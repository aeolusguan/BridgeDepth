"""Original image API and numerics, using a small real DINOv2/DPT image model."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("cv2")
pytest.importorskip("imageio")
pytest.importorskip("huggingface_hub")

from bridgedepth.bridgedepth import BridgeDepth
from bridgedepth.config import export_model_config, get_cfg
from test_repository_integration import assert_tree_close


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def image_model():
    torch.manual_seed(73)
    cfg = get_cfg()
    cfg.BRIDGEDEPTH.ENCODER_VIT = "vits"
    cfg.BACKBONE.OUT_CHANNELS = 32
    cfg.DPN.MAX_DISP = 64
    cfg.NMP.NUM_PROP_LAYERS = 1
    cfg.NMP.NUM_INFER_LAYERS = 2
    cfg.NMP.NUM_REFINE_LAYERS = 2
    return BridgeDepth(cfg), cfg


def test_original_image_checkpoint_layout_and_predictions(image_model):
    model, _ = image_model
    reference = torch.load(
        Path(__file__).parent / "fixtures/image_before_integration.pt", weights_only=True)
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == reference["parameter_shapes"]
    assert model.stereo_encoder.backbone.normalize_input
    with torch.no_grad():
        for mode in ("eval", "train"):
            model.train(mode == "train")
            output = model({"img1": reference["left"].clone(), "img2": reference["right"].clone()})
            assert_tree_close(output, reference["outputs"][mode])


def test_original_from_pretrained_and_config_factories(image_model, tmp_path):
    model, cfg = image_model
    checkpoint = tmp_path / "image.pth"
    torch.save({"model_config": export_model_config(cfg), "model": model.state_dict()}, checkpoint)
    restored = BridgeDepth.from_pretrained(checkpoint)
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)

    from bridgedepth.stereo import build_criterion, build_model
    from bridgedepth.stereo.DPN import DPN
    from bridgedepth.stereo.NMRF import NMRF

    assert isinstance(build_model(cfg), NMRF)
    assert isinstance(DPN(cfg), DPN)
    assert callable(build_criterion)


@pytest.mark.parametrize("name", ["SwinDPT", "ConvNextDPT"])
def test_original_alternative_backbones_without_downloads(name):
    pytest.importorskip("torchvision")
    from bridgedepth.stereo import backbone

    model = getattr(backbone, name)(output_dim=16, _init_weights=False).eval()
    with torch.no_grad():
        features = model(torch.rand(1, 3, 64, 96) * 255)
    assert [tuple(feature.shape) for feature in features] == [(1, 16, 16, 24), (1, 16, 8, 12)]
