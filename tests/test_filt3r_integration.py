"""Bundled-architecture checks; no external checkout, checkpoint or GPU needed."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from bridgedepth.models import FrozenFILT3R


@pytest.fixture(scope="module")
def native():
    return importlib.import_module("bridgedepth.models._filt3r.model")


def config_args(head_type="linear"):
    return dict(
        img_size=(32, 48), patch_size=16, enc_embed_dim=64, enc_depth=1,
        enc_num_heads=4, dec_embed_dim=64, dec_depth=4, dec_num_heads=4,
        state_dec_num_heads=4, state_size=4, local_mem_size=4, ray_enc_depth=1,
        pos_embed="RoPE100", output_mode="pts3d+pose", head_type=head_type,
        pose_head=True, landscape_only=False, model_update_type="filt3r",
    )


def make_native(native, head_type="linear"):
    torch.manual_seed(19)
    return native.ARCroco3DStereo(native.ARCroco3DStereoConfig(**config_args(head_type))).eval()


def views_from_frames(frames):
    batch, _, height, width = frames[0].shape
    return [dict(
        img=frame, true_shape=torch.tensor([[height, width]]).expand(batch, 2),
        img_mask=torch.ones(batch, dtype=torch.bool),
        ray_mask=torch.zeros(batch, dtype=torch.bool),
        ray_map=torch.zeros(batch, height, width, 6),
        reset=torch.zeros(batch, dtype=torch.bool),
        update=torch.ones(batch, dtype=torch.bool),
    ) for frame in frames]


@pytest.mark.parametrize("head_type", ["linear", "dpt"])
def test_native_sequence_matches_streaming_outputs_and_states(native, head_type):
    backbone = make_native(native, head_type)
    model = FrozenFILT3R(backbone)
    frames = torch.randn(3, 2, 3, 32, 48)
    with torch.no_grad():
        predictions, _, native_states = backbone._forward_impl(views_from_frames(frames), ret_state=True)
        state = None
        for index, frame in enumerate(frames):
            output, state = model(frame, state)
            for key in predictions[index]:
                torch.testing.assert_close(output.prediction[key], predictions[index][key], rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(state.state_feat, native_states[index + 1][0], rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(state.mem, native_states[index + 1][3], rtol=2e-4, atol=2e-5)
        camera_utils = importlib.import_module("bridgedepth.models._filt3r.utils.camera")
        torch.testing.assert_close(output.camera_pose, camera_utils.pose_encoding_to_camera(output.prediction["camera_pose"]))


@pytest.mark.parametrize("head_type", ["linear", "dpt"])
def test_native_demo_matches_streaming_predictions_and_filter(native, monkeypatch, head_type):
    backbone = make_native(native, head_type)
    model = FrozenFILT3R(backbone)
    frames = torch.randn(5, 2, 3, 32, 48)
    filter_outputs = []
    native_filter = backbone._compute_kalman_ema_gain_and_cov

    def record_filter(*args, **kwargs):
        gain, covariance, statistics = native_filter(*args, **kwargs)
        filter_outputs.append(tuple(value.detach().clone()
                                    for value in (gain, covariance, statistics["delta_ema"])))
        return gain, covariance, statistics

    monkeypatch.setattr(backbone, "_compute_kalman_ema_gain_and_cov", record_filter)
    with torch.no_grad():
        predictions, _ = backbone.forward_recurrent_lighter(views_from_frames(frames), device="cpu")
        native_filter_outputs = list(filter_outputs)
        filter_outputs.clear()
        state = None
        for frame, prediction in zip(frames, predictions):
            output, state = model.step(frame, state)
            assert output.prediction.keys() == prediction.keys()
            for key in prediction:
                torch.testing.assert_close(output.prediction[key], prediction[key], rtol=2e-4, atol=2e-5)

    assert len(predictions) == len(frames)
    assert len(filter_outputs) == len(native_filter_outputs) == len(frames) - 1
    for actual, expected in zip(filter_outputs, native_filter_outputs):
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(actual_value, expected_value, rtol=2e-4, atol=2e-5)


def test_native_dpt_backward_reaches_inputs_and_previous_frame(native):
    model = FrozenFILT3R(make_native(native, "dpt")).train()
    image = torch.randn(1, 3, 32, 48, requires_grad=True)
    _, state = model(image)
    state.state_feat.retain_grad()
    output, _ = model(image * 0.7, state)
    loss = output.depth.square().mean() + output.camera_pose[:, :3, 3].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert state.state_feat.grad is not None and state.state_feat.grad.abs().sum() > 0
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_native_checkpoint_round_trip(native, tmp_path):
    backbone = make_native(native)
    arguments = ", ".join(f"{key}={value!r}" for key, value in config_args().items())
    expression = f"ARCroco3DStereo(ARCroco3DStereoConfig({arguments}))"
    checkpoint = tmp_path / "tiny.pth"
    torch.save({"args": SimpleNamespace(model=expression), "model": backbone.state_dict()}, checkpoint)
    loaded = FrozenFILT3R.from_pretrained(checkpoint)
    image = torch.randn(1, 3, 32, 48)
    with torch.no_grad():
        expected, _ = FrozenFILT3R(backbone)(image)
        actual, _ = loaded(image)
    torch.testing.assert_close(actual.depth, expected.depth)
    torch.testing.assert_close(actual.camera_pose, expected.camera_pose)
