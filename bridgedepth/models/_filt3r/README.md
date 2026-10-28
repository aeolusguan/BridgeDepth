# Vendored FILT3R inference code

This directory contains the inference dependency closure of
[jinotter3/FILT3R](https://github.com/jinotter3/FILT3R), pinned to commit
[`6cf6d28b6ec7fbcc985187a0f099ccbc90224392`](https://github.com/jinotter3/FILT3R/tree/6cf6d28b6ec7fbcc985187a0f099ccbc90224392).
It includes FILT3R's inherited CUT3R, DUSt3R and CroCo components. It does not
include datasets, training launchers, demos, losses, or pretrained weights.

## Source mapping

- `model.py`, `blocks.py`, `patch_embed.py`, `heads/` and `utils/` come from
  upstream `src/dust3r/` with the same paths.
- `utils/geometry.py` contains only the original `geotrf` function and its
  copyright header; unused geometry utilities and their SciPy dependency are omitted.
- `croco/` contains upstream `src/croco/models/{croco,blocks,pos_embed,masking,dpt_block}.py`
  and `src/croco/models/curope/`.
- `LICENSE` is the unmodified upstream root license. `croco/LICENSE` and
  `croco/NOTICE` are unmodified copies of upstream `src/croco/LICENSE` and `NOTICE`.

## Local changes

Imports are relative to this private package. Upstream path injection, executable
module demonstrations, unused imports, and the import-time TF32 setting are removed.
The unused Accelerate logger dependency is replaced with Python logging.
The pose encoder imports its postprocessing helper when called, avoiding a
circular import when using the camera conversion utilities directly.
`ARCroco3DStereo`, `ARCroco3DStereoConfig`, model methods and parameter names remain
compatible with the native implementation. The upstream model's training methods
are retained for compatibility. The BridgeDepth wrapper manages frozen inference
and explicit streaming state.

`croco.pos_embed.RoPE2DPyTorch` fixes upstream's negative-position lookup error:
the pose token uses position -1, cosine is even, and sine is odd. It also applies
the configured `F0` consistently with the CUDA implementation. `RoPE2D` selects
the optional kernel for supported CUDA dtypes and otherwise uses this fallback;
CPU and bfloat16 always use PyTorch. No parameters are added to the state dict.

The CUDA extension is renamed `_bridgedepth_filt3r_curope` and imported relatively,
so it cannot select an unrelated global `curope` module. Its launcher uses the
current CUDA stream with a device guard and launch-error check; dispatch uses
`tokens.scalar_type()` for current PyTorch compatibility. Its numerical kernel
and native autograd wrapper are otherwise unchanged. CUDA compilation and runtime
behavior still require validation on a CUDA machine; CPU tests do not establish
CUDA performance or correctness.

## Optional CUDA build

With a CUDA-enabled PyTorch installation and a matching CUDA toolkit, run from
the repository root:

```sh
cd bridgedepth/models/_filt3r/croco/curope
python setup.py build_ext --inplace
```

`TORCH_CUDA_ARCH_LIST` can restrict the build architectures. Rebuild after changing
PyTorch or CUDA versions. No extension compilation occurs during ordinary imports.

## License notices

Keep the supplied license files and per-file copyright headers when redistributing
this subset. The upstream root license contains both CUT3R's CC BY-NC-SA 4.0 notice
and an MIT notice for Inception3D contributions. CroCo and inherited Naver files
carry CC BY-NC-SA 4.0 restrictions, and `croco/NOTICE` lists additional upstream
subcomponent licenses. Vendoring does not relicense these components under a
project-wide license. Pretrained checkpoints have their own upstream terms.
