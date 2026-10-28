"""Regression against the streaming implementation captured before integration."""

from dataclasses import fields, is_dataclass
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest
import torch

from bridgedepth.models import StreamingBridgeDepth, StreamingBridgeDepthOutput
from test_bridge import make_bridge


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_tree_close(actual, expected):
    if is_dataclass(actual):
        actual = {field.name: getattr(actual, field.name) for field in fields(actual)}
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_tree_close(actual[name], expected[name])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            assert_tree_close(result, reference)
    elif isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        assert actual == expected


def test_preintegration_parameters_predictions_and_recurrent_state():
    reference = torch.load(
        Path(__file__).parent / "fixtures/streaming_before_integration.pt", weights_only=True)
    model = make_bridge("dpt").eval()
    assert isinstance(model, StreamingBridgeDepth)
    torch.manual_seed(reference["provenance"]["input_seed"])
    for adapter in model.adapters:
        torch.nn.init.normal_(adapter.to_filt.weight, std=0.02)
        torch.nn.init.normal_(adapter.to_filt.bias, std=0.01)
    assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == reference["parameter_shapes"]
    state = None
    with torch.no_grad():
        for left, right, frame in zip(reference["left"], reference["right"], reference["frames"]):
            output, state = model(left, right, state)
            assert isinstance(output, StreamingBridgeDepthOutput)
            assert_tree_close(output, frame["output"])
            assert_tree_close(state, frame["state"])
        model.train()
        output, state = model(reference["left"][0], reference["right"][0])
    assert_tree_close(output, reference["training"]["output"])
    assert_tree_close(state, reference["training"]["state"])


def test_streaming_uses_shared_components_and_preserves_normalization():
    from bridgedepth.blocks import Alignment
    from bridgedepth.stereo.NMRF import NMRF
    from bridgedepth.stereo.backbone import Backbone

    model = make_bridge()
    assert type(model.align_0) is type(model.align_1) is Alignment
    assert type(model.stereo_encoder) is NMRF
    assert type(model.stereo_encoder.backbone) is Backbone
    assert not model.stereo_encoder.backbone.normalize_input
    raw = Backbone(output_dim=32).eval()
    raw.load_state_dict(model.stereo_encoder.backbone.state_dict(), strict=True)
    image = torch.rand(2, 3, 32, 48) * 255
    with torch.no_grad():
        original_features = raw(image)
        streaming_features = model.stereo_encoder.backbone(2 * (image / 255) - 1)
    for actual, expected in zip(streaming_features, original_features):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["image", "streaming"])
def test_model_imports_keep_optional_dependencies_isolated(mode):
    if mode == "image" and any(importlib.util.find_spec(name) is None
                               for name in ("cv2", "imageio", "huggingface_hub")):
        pytest.skip("image inference dependencies are not installed")
    if mode == "image":
        blocked = ("transformers", "bridgedepth.models", "torchvision")
        statement = "from bridgedepth.bridgedepth import BridgeDepth"
    else:
        blocked = ("transformers", "torchvision", "cv2", "imageio", "bridgedepth.monocular")
        statement = "from bridgedepth.models import FrozenFILT3R, StreamingBridgeDepth"
    program = textwrap.dedent(f"""
        import importlib.abc
        import sys
        import torch
        blocked = {blocked!r}
        class RejectOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
                    raise ImportError('Unexpected optional import: ' + fullname)
        sys.meta_path.insert(0, RejectOptional())
        original_path = tuple(sys.path)
        {statement}
        assert tuple(sys.path) == original_path
        assert not any(key == name or key.startswith(name + '.')
                       for name in blocked for key in sys.modules)
    """)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
                            env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def test_streaming_dataset_sources_are_not_git_ignored():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists() or shutil.which("git") is None:
        pytest.skip("requires a Git checkout")
    sources = [str(path.relative_to(root)) for path in (root / "bridgedepth/datasets").glob("*.py")]
    assert len(sources) == 5
    result = subprocess.run(["git", "check-ignore", "--no-index", *sources], cwd=root,
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stdout + result.stderr
