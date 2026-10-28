"""Streaming behavior with a controlled backbone and FILT3R's native filter."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bridgedepth.models import FrozenFILT3R
from bridgedepth.models._filt3r.model import ARCroco3DStereo


class FakeMemory(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = nn.Parameter(torch.zeros(1, 1, 4))
        self.projection = nn.Linear(4, 4)

    def inquire(self, feature, memory):
        return self.projection(feature + memory)

    def update_mem(self, memory, feature, pose):
        return memory * 0.5 + feature * 0.25 + pose * 0.25


class FakeBackbone(nn.Module):
    pose_head_flag = True
    dec_depth = 4
    _ema_update = ARCroco3DStereo._ema_update
    _compute_kalman_ema_gain_and_cov = ARCroco3DStereo._compute_kalman_ema_gain_and_cov

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.decoder = nn.Linear(4, 4)
        self.points_head = nn.Linear(4, 3)
        self.pose_token = nn.Parameter(torch.zeros(1, 1, 4))
        self.pose_retriever = FakeMemory()
        self.dropout = nn.Dropout(0.75)
        self.config = SimpleNamespace(kalman_p_init=1.5, kalman_q_min=0.2,
                                      kalman_q_max=0.2, kalman_fixed_r=1.0,
                                      kalman_ema_beta_delta=0.25)

    def _get_hparam(self, name, default):
        return getattr(self.config, name, default)

    def _encode_image(self, image, shape):
        tokens = self.dropout(self.encoder(image.flatten(2).transpose(1, 2)))
        positions = torch.zeros(*tokens.shape[:2], 2, device=image.device, dtype=torch.long)
        return [tokens], positions, None

    def _init_state(self, tokens, positions):
        return tokens.new_zeros(len(tokens), 2, 4), positions[:, :2]

    def _get_img_level_feat(self, tokens):
        return tokens.mean(dim=1, keepdim=True)

    def _decoder(self, state, state_positions, tokens, positions, pose,
                 pose_positions, return_attn):
        context = tokens.mean(dim=1, keepdim=True)
        candidate = self.decoder(state + context)
        patches = self.decoder(tokens + state.mean(dim=1, keepdim=True))
        camera = pose + context + state.mean(dim=1, keepdim=True)
        decoded = [tokens] + [torch.cat((camera, patches), dim=1)] * 4
        return ([candidate], decoded), (None, None, None, None)

    def _downstream_head(self, inputs, shape, **kwargs):
        height, width = (int(x) for x in shape[0])
        points = self.points_head(inputs[-1][:, 1:]).reshape(-1, height, width, 3)
        translation = inputs[-1][:, 0, :3]
        quaternion = translation.new_tensor([1, 0, 0, 0]).expand(len(translation), 4)
        return {"pts3d_in_self_view": points,
                "camera_pose": torch.cat((translation, quaternion), dim=1)}


def make_model():
    torch.manual_seed(4)
    return FrozenFILT3R(FakeBackbone())


def test_backbone_stays_frozen_and_eval_when_outer_model_trains():
    model = make_model()
    outer = nn.ModuleDict({"monocular": model, "stereo": nn.Linear(3, 4)})
    outer.eval()
    assert not outer["stereo"].training
    outer.train()
    assert outer["stereo"].training
    assert outer["stereo"].weight.requires_grad
    assert all(not module.training for module in model.backbone.modules())
    assert all(not p.requires_grad for p in model.parameters())
    image = torch.randn(2, 3, 2, 3)
    first, _ = model(image)
    second, _ = model(image)
    torch.testing.assert_close(first.depth, second.depth)
    assert first.depth.shape == (2, 1, 2, 3)
    assert first.camera_pose.shape == (2, 4, 4)
    torch.testing.assert_close(first.depth[:, 0], first.prediction["pts3d_in_self_view"][..., 2])
    torch.testing.assert_close(first.camera_pose[:, :3, 3], first.prediction["camera_pose"][:, :3])


def test_first_frame_full_write_then_kalman_covariance_and_ema_continue():
    model = make_model()
    image = torch.randn(2, 3, 2, 3)
    _, first = model(image)
    torch.testing.assert_close(first.state_feat, first.prev_candidate_state_feat)
    torch.testing.assert_close(first.state_cov, torch.full_like(first.state_cov, 1.5))
    assert first.kalman_stats is None
    _, second = model(image + 0.2, first)
    gain = 1.7 / (2.7 + 1e-6)
    expected = gain * second.prev_candidate_state_feat + (1 - gain) * first.state_feat
    torch.testing.assert_close(second.state_feat, expected)
    expected_cov = (1 - gain)**2 * 1.7 + gain**2
    torch.testing.assert_close(second.state_cov, torch.full_like(second.state_cov, expected_cov))
    delta = (second.prev_candidate_state_feat - first.prev_candidate_state_feat).norm(dim=-1).mean(dim=1)
    torch.testing.assert_close(second.kalman_stats["delta_ema"], delta.clamp_min(1e-3))
    old_ema = second.kalman_stats["delta_ema"].clone()
    _, third = model(image - 0.3, second)
    delta3 = (third.prev_candidate_state_feat - second.prev_candidate_state_feat).norm(dim=-1).mean(dim=1)
    torch.testing.assert_close(third.kalman_stats["delta_ema"], (0.75 * old_ema + 0.25 * delta3).clamp_min(1e-3))
    torch.testing.assert_close(second.kalman_stats["delta_ema"], old_ema)
    assert third.frame_index == 3


def test_frozen_decoder_preserves_gradients_into_inputs_and_history():
    model = make_model()
    image = torch.randn(2, 3, 2, 3, requires_grad=True)
    _, state = model(image)
    state.state_feat.retain_grad()
    output, _ = model(image + 0.5, state)
    (output.depth.square().mean() + output.camera_pose[:, :3, 3].square().mean()).backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert state.state_feat.grad is not None and state.state_feat.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.parameters())


def test_autocast_preserves_native_float32_state_write_masks():
    model = make_model()
    image = torch.randn(2, 3, 2, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _, state = model(image)
        assert state.prev_candidate_state_feat.dtype == torch.bfloat16
        assert state.state_cov.dtype == torch.bfloat16
        assert state.state_feat.dtype == torch.float32
        assert state.mem.dtype == torch.float32
        _, next_state = model(image, state)
        assert next_state.state_feat.dtype == torch.float32


def test_detach_preserves_stream_and_new_clip_is_independent():
    model = make_model()
    frames = torch.randn(4, 2, 3, 2, 3, requires_grad=True)
    _, first = model(frames[0])
    _, state = model(frames[1], first)
    assert state.state_feat.requires_grad
    detached = state.detach()
    for key, value in vars(detached).items():
        original = getattr(state, key)
        if isinstance(value, torch.Tensor):
            assert value.grad_fn is None and not value.requires_grad
            torch.testing.assert_close(value, original)
    assert not detached.kalman_stats["delta_ema"].requires_grad
    continuous, next_state = model(frames[2], state)
    chunked, chunk_state = model(frames[2], detached)
    torch.testing.assert_close(continuous.depth, chunked.depth)
    torch.testing.assert_close(next_state.state_cov, chunk_state.state_cov)
    prefix_before = state.state_feat.clone()
    model(frames[3], next_state)
    torch.testing.assert_close(state.state_feat, prefix_before)
    repeated, _ = model(frames[2], state)
    torch.testing.assert_close(repeated.depth, continuous.depth)
    reset, reset_state = model(frames[2], None)
    assert reset_state.frame_index == 1
    assert not torch.allclose(reset.depth, continuous.depth)


def test_invalid_input_and_partial_reset_are_rejected():
    model = make_model()
    image = torch.randn(2, 3, 2, 3)
    _, state = model(image)
    with pytest.raises(ValueError, match="batch size"):
        model(image[:1], state)
    with pytest.raises(TypeError, match="reset"):
        model.step(image, state, reset=torch.tensor([True, False]))
