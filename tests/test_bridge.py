"""Real stereo/decoder checks with small randomly initialized CPU models."""

from dataclasses import replace

import pytest
import torch

from bridgedepth.stereo.backbone import Backbone
from bridgedepth.blocks import Alignment, CrossAttention
from bridgedepth.stereo.DPN import DPN
from bridgedepth.stereo.NMRF import NMRF
from bridgedepth.models import StreamingBridgeDepth, FrozenFILT3R
from bridgedepth.models._filt3r.model import ARCroco3DStereo, ARCroco3DStereoConfig


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("stride,shape", [(1, (5, 7)), (0.5, (6, 10)), (2, (3, 5))])
def test_primitives_cross_resolution_padding_and_backward(stride, shape):
    height, width = shape
    attention = CrossAttention(16, 16, 16, (4, 4), stride, 4)
    query = torch.randn(2, height, width, 2, 16, requires_grad=True)
    context = torch.randn(2, int(height * stride), int(width * stride), 1, 16,
                          requires_grad=True)
    result = attention(query, context)
    assert result.shape == query.shape
    result.square().mean().backward()
    assert torch.isfinite(result).all()
    assert query.grad.abs().sum() > 0
    assert context.grad.abs().sum() > 0


def test_primitives_stereo_proposals_and_bidirectional_alignment():
    torch.manual_seed(3)
    dpn = DPN(cost_group=4, num_proposals=2, feat_dim=32, prior_dim=16,
              num_prop_layers=1, prop_embed_dim=16, mlp_ratio=4,
              split_size=1, prop_n_heads=4)
    stereo = NMRF(Backbone(output_dim=32, normalize_input=False), dpn, num_proposals=2,
                  max_disp=64, infer_embed_dim=16)
    prior = torch.randn(2, 4, 6, 16, requires_grad=True)
    result = stereo({"img1": torch.randn(2, 3, 32, 48),
                     "img2": torch.randn(2, 3, 32, 48), "depth_prior": prior})
    assert result["proposal"].shape == (2, 24, 2)
    assert result["prob"].shape == (48, 8)
    hypothesis, positions = stereo.construct_hypothesis_embedding(
        result["proposal"].flatten(0, 1).detach(),
        result["fmap1_concat_list"][0], result["fmap2_concat_list"][0],
        result["fmap1_gw_list"][0], result["fmap2_gw_list"][0])
    assert hypothesis.shape == (2, 4, 6, 2, 16)
    assert positions.shape == (2, 4, 6, 2, 31)
    alignment = Alignment(2, 16, window_size=6, num_heads=4,
                          return_intermediate=True)
    hypotheses, monocular = alignment(hypothesis, prior, positions)
    assert hypotheses.shape == (2, 2, 4, 6, 2, 16)
    assert monocular.shape == (2, 2, 4, 6, 16)
    loss = monocular.square().mean() + result["proposal"].square().mean()
    loss.backward()
    assert prior.grad.abs().sum() > 0
    assert stereo.backbone.conv1.weight.grad.abs().sum() > 0
    assert stereo.dpn.prop_head.layers[-1].weight.grad.abs().sum() > 0


def make_bridge(head_type="linear"):
    torch.manual_seed(19)
    config = ARCroco3DStereoConfig(
        img_size=(32, 48), patch_size=16, enc_embed_dim=64, enc_depth=1,
        enc_num_heads=4, dec_embed_dim=32, dec_depth=12, dec_num_heads=4,
        state_dec_num_heads=4, state_size=4, local_mem_size=4, ray_enc_depth=1,
        pos_embed="RoPE100", output_mode="pts3d+pose", head_type=head_type,
        pose_head=True, landscape_only=False, model_update_type="filt3r")
    monocular = FrozenFILT3R(ARCroco3DStereo(config))
    return StreamingBridgeDepth(monocular, embed_dim=16, stereo_feature_dim=32,
                         max_disparity=64, proposal_layers=1)


@pytest.mark.parametrize("head_type", ["linear", "dpt"])
def test_native_predictions_and_recurrence_match_with_zero_bridges(head_type):
    model = make_bridge(head_type).eval()
    reference_state = bridged_state = None
    with torch.no_grad():
        for _ in range(3):
            left, right = torch.randn(2, 3, 32, 48), torch.randn(2, 3, 32, 48)
            reference, reference_state = model.monocular(left, reference_state)
            output, bridged_state = model(left, right, bridged_state)
            for key in reference.prediction:
                torch.testing.assert_close(output.prediction[key], reference.prediction[key])
            for name in ("state_feat", "mem", "state_cov", "prev_candidate_state_feat"):
                torch.testing.assert_close(getattr(bridged_state, name), getattr(reference_state, name))
            if reference_state.kalman_stats is not None:
                for key in reference_state.kalman_stats:
                    torch.testing.assert_close(bridged_state.kalman_stats[key], reference_state.kalman_stats[key])
            assert bridged_state.frame_index == reference_state.frame_index
            assert output.disparity.shape == (2, 1, 32, 48)
            assert output.depth.shape == (2, 1, 32, 48)
            assert output.camera_pose.shape == (2, 4, 4)
            assert torch.isfinite(output.disparity).all()
            assert "aux_outputs" not in output.stereo


def test_schedule_has_six_coarse_and_six_fine_exchanges_before_native_dual_layers(monkeypatch):
    model = make_bridge().eval()
    backbone = model.monocular.backbone
    events, snapshots, handles = [], {}, []

    def alignment_hook(index):
        def record(module, args):
            events.append(("alignment", index))
            expected = (4, 6, 2) if index < 6 else (8, 12, 1)
            assert args[0].shape[1:4] == expected
            assert args[1].shape[1:4] == (4, 6, 1)
        return record

    def state_hook(index):
        def record(module, args):
            events.append(("state", index))
            snapshots[index] = args[:2]
        return record

    def image_hook(index):
        def record(module, args):
            events.append(("image", index))
            previous_scene, corrected_image = snapshots[index]
            assert args[0] is corrected_image
            assert args[1] is previous_scene
        return record

    for index, layer in enumerate(list(model.align_0.layers) + list(model.align_1.layers)):
        handles.append(layer.register_forward_pre_hook(alignment_hook(index)))
        handles.append(backbone.dec_blocks_state[index].register_forward_pre_hook(state_hook(index)))
        handles.append(backbone.dec_blocks[index].register_forward_pre_hook(image_hook(index)))
    filters, writes = [], []
    original_filter, original_write = model.monocular._filter, backbone.pose_retriever.update_mem

    def record_filter(*args):
        filters.append(1)
        return original_filter(*args)

    def record_write(*args):
        writes.append(1)
        return original_write(*args)

    monkeypatch.setattr(model.monocular, "_filter", record_filter)
    monkeypatch.setattr(backbone.pose_retriever, "update_mem", record_write)
    with torch.no_grad():
        output, state = model(torch.randn(1, 3, 32, 48), torch.randn(1, 3, 32, 48))
    for handle in handles:
        handle.remove()
    assert events == [(kind, index) for index in range(12) for kind in ("alignment", "state", "image")]
    assert len(filters) == len(writes) == state.frame_index == 1
    assert output.stereo["coarse_disp"].shape == (1, 32, 48, 2)
    assert output.stereo["proposal"].shape == (1, 24, 2)


def test_training_outputs_optimizer_step_and_frozen_monocular_weights():
    model = make_bridge("dpt").train()
    assert all(not module.training for module in model.monocular.backbone.modules())
    assert model.stereo_encoder.training and model.align_0.training
    frozen_weight = model.monocular.backbone.patch_embed.proj.weight.detach().clone()
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    output, state = model(torch.randn(2, 3, 32, 48), torch.randn(2, 3, 32, 48))
    auxiliary = output.stereo["aux_outputs"]
    assert len(auxiliary) == 10
    assert sum("disp_pred" in item for item in auxiliary) == 5
    assert sum("coarse_disp" in item for item in auxiliary) == 5
    # Depth and disparity supervision suffice; no pose labels or pose loss.
    loss = output.depth.square().mean() + output.disparity.square().mean()
    loss += output.stereo["proposal"].square().mean() + output.stereo["prob"].square().mean()
    loss += output.stereo["coarse_disp"].square().mean() + output.stereo["logit"].square().mean()
    loss += sum(item.get("disp_pred", item.get("coarse_disp")).square().mean() for item in auxiliary)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.adapters[-1].to_filt.weight.grad.abs().sum() > 0
    assert model.stereo_encoder.backbone.conv1.weight.grad.abs().sum() > 0
    assert model.stereo_encoder.dpn.prop_head.layers[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in model.monocular.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert model.adapters[-1].to_filt.weight.detach().abs().sum() > 0
    torch.testing.assert_close(model.monocular.backbone.patch_embed.proj.weight, frozen_weight)
    detached = state.detach()
    assert not detached.state_feat.requires_grad
    assert not detached.mem.requires_grad


def test_last_bridge_writes_stereo_into_scene_state_and_future_predictions():
    model = make_bridge("dpt").eval()
    torch.nn.init.normal_(model.adapters[-1].to_filt.weight, std=0.03)
    left = torch.randn(1, 3, 32, 48)
    right = torch.randn(1, 3, 32, 48, requires_grad=True)
    first, state = model(left, right)
    with torch.no_grad():
        changed, changed_state = model(left, right + 0.3)
    assert (first.depth.detach() - changed.depth).abs().max() > 1e-8
    assert (first.camera_pose.detach() - changed.camera_pose).abs().max() > 1e-8
    assert (state.state_feat.detach() - changed_state.state_feat).abs().max() > 1e-8
    # Cut pose-memory history to isolate the scene-state temporal path.
    state = replace(state, mem=state.mem.detach())
    state.state_feat.retain_grad()
    second, _ = model(left * 0.9, right.detach() * 0.8, state)
    stereo_history_grad = torch.autograd.grad(second.disparity.mean(), state.state_feat,
                                               retain_graph=True)[0]
    assert torch.isfinite(stereo_history_grad).all() and stereo_history_grad.abs().sum() > 0
    (second.depth.square().mean() + second.camera_pose[:, :3, 3].square().mean()).backward()
    assert state.state_feat.grad.abs().sum() > 0
    assert right.grad is not None and torch.isfinite(right.grad).all()
    assert right.grad.abs().sum() > 0
    assert model.align_1.layers[-1].update.attn.proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.monocular.parameters())


def test_coarse_regression_keeps_proposal_coordinates_detached():
    model = make_bridge()
    features = torch.randn(1, 4, 6, 2, 16, requires_grad=True)
    proposal = torch.full((1, 24, 2), 2.0, requires_grad=True)
    coarse, logit = model._coarse_prediction(features, proposal)
    feature_grad, proposal_grad = torch.autograd.grad(
        coarse.mean() + logit.square().mean(), (features, proposal), allow_unused=True)
    assert feature_grad is not None and feature_grad.abs().sum() > 0
    assert proposal_grad is None


def test_coarse_to_fine_disparity_units_winner_and_lower_median():
    model = make_bridge().eval()
    for head in (model.infer_head, model.refine_head):
        torch.nn.init.zeros_(head.layers[-1].weight)
        torch.nn.init.zeros_(head.layers[-1].bias)
    with torch.no_grad():
        stereo = model.stereo_encoder({"img1": torch.randn(1, 3, 32, 48),
                                       "img2": torch.randn(1, 3, 32, 48),
                                       "depth_prior": torch.randn(1, 4, 6, 16)})
        coarse, _ = model._coarse_prediction(torch.randn(1, 4, 6, 2, 16),
                                             torch.full((1, 24, 2), 2.0))
        torch.testing.assert_close(coarse, torch.full_like(coarse, 16.0))
    pattern = torch.arange(16).reshape(4, 4).repeat(8, 12).float()[None]
    coarse = torch.stack((pattern * 4, pattern * 4 + 4), dim=-1).requires_grad_()
    logits = torch.stack((torch.zeros_like(pattern), torch.ones_like(pattern)), dim=-1)
    seed, features, _ = model._fine_hypothesis(coarse, logits, stereo)
    assert not seed.requires_grad
    torch.testing.assert_close(seed, torch.full_like(seed, 8.0))
    fine = model._fine_prediction(features, seed)
    torch.testing.assert_close(fine, torch.full_like(fine, 32.0))


def test_default_stereo_configuration_and_non_window_multiple_resolution():
    monocular = make_bridge().monocular
    model = StreamingBridgeDepth(monocular).eval()
    with torch.no_grad():
        output, state = model(torch.randn(1, 3, 48, 80), torch.randn(1, 3, 48, 80))
    assert output.disparity.shape == output.depth.shape == (1, 1, 48, 80)
    assert output.stereo["proposal"].shape == (1, 60, 2)
    assert output.stereo["prob"].shape == (60, 40)
    assert state.frame_index == 1


def test_mismatched_stereo_images_and_non_patch_aligned_sizes_are_rejected():
    model = make_bridge()
    with pytest.raises(ValueError, match="matching"):
        model(torch.randn(1, 3, 32, 48), torch.randn(2, 3, 32, 48))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 3, 34, 48), torch.randn(1, 3, 34, 48))
