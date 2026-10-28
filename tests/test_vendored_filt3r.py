"""Vendored import isolation and signed-coordinate native PyTorch RoPE checks."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import torch


def test_vendored_imports_ignore_generic_modules_and_preserve_search_path():
    # Run in a fresh interpreter: an earlier test must not hide an accidental
    # absolute import behind an already-cached vendored module.
    program = textwrap.dedent("""
        import importlib
        import sys
        import types
        import torch, transformers, einops, numpy
        from transformers import PreTrainedModel, PretrainedConfig

        generic_names = ('dust3r', 'models', 'croco')
        existing = {name: types.ModuleType(name) for name in generic_names}
        sys.modules.update(existing)
        original_path = tuple(sys.path)
        original_modules = set(sys.modules)

        model = importlib.import_module('bridgedepth.models._filt3r.model')
        rope = importlib.import_module('bridgedepth.models._filt3r.croco.pos_embed')

        assert model.ARCroco3DStereo.__module__.startswith('bridgedepth.models._filt3r.')
        assert model.ARCroco3DStereoConfig.__module__.startswith('bridgedepth.models._filt3r.')
        assert rope.RoPE2DPyTorch.__module__ == rope.__name__
        assert tuple(sys.path) == original_path, 'vendored import changed sys.path'
        assert all(sys.modules[name] is module for name, module in existing.items())
        imported = set(sys.modules) - original_modules
        assert not any(name.startswith(prefix + '.')
                       for name in imported for prefix in generic_names), imported
    """)
    environment = os.environ.copy()
    environment.pop("FILT3R_REPO", None)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=Path(__file__).resolve().parents[1],
        env=environment, text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def complex_rope_reference(tokens, positions, base=100.0, initial_frequency=1.0):
    """Independent complex-plane formulation of the two axial rotations."""
    axis_width = tokens.shape[-1] // 2
    frequencies = initial_frequency * base ** (
        -torch.arange(axis_width // 2, device=tokens.device, dtype=torch.float64)
        / (axis_width // 2)
    )
    rotated_axes = []
    for axis, values in enumerate(tokens.split(axis_width, dim=-1)):
        real, imaginary = values.chunk(2, dim=-1)
        angles = positions[..., axis, None].to(torch.float64) * frequencies
        rotation = torch.polar(torch.ones_like(angles), angles)[:, None]
        rotated = torch.complex(real, imaginary) * rotation
        rotated_axes.append(torch.cat((rotated.real, rotated.imag), dim=-1))
    return torch.cat(rotated_axes, dim=-1).to(tokens.dtype)


def test_native_rope_signed_positions_match_complex_rotation():
    from bridgedepth.models._filt3r.croco.pos_embed import RoPE2DPyTorch

    torch.manual_seed(12)
    tokens = torch.randn(2, 3, 4, 16, dtype=torch.float64)
    # FILT3R uses (-1,-1) for pose tokens; other negatives also exercise the
    # formula rather than treating -1 as a special-case identity embedding.
    positions = torch.tensor([
        [[-1, -1], [0, 0], [2, 3], [-3, 2]],
        [[1, -2], [-1, 4], [0, -1], [3, 0]],
    ])
    rope = RoPE2DPyTorch(freq=100.0, F0=1.0)
    actual = rope(tokens, positions)
    expected = complex_rope_reference(tokens, positions)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(rope(actual, -positions), tokens, atol=2e-7, rtol=2e-7)
    # Zero positions are exactly unchanged, including both axial halves.
    torch.testing.assert_close(actual[0, :, 1], tokens[0, :, 1], atol=0, rtol=0)


def test_native_rope_negative_pose_tokens_support_autograd():
    from bridgedepth.models._filt3r.croco.pos_embed import RoPE2DPyTorch

    torch.manual_seed(13)
    tokens = torch.randn(1, 2, 3, 16, dtype=torch.float64, requires_grad=True)
    positions = torch.tensor([[[-1, -1], [0, 2], [3, -2]]])
    rope = RoPE2DPyTorch(freq=100.0, F0=1.0)
    assert torch.autograd.gradcheck(lambda value: rope(value, positions), (tokens,))
