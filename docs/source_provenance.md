# Source provenance

## BridgeDepth image model and shared stereo components

The repository is based on [aeolusguan/BridgeDepth](https://github.com/aeolusguan/BridgeDepth), commit `9a231601e6981d3536fa6a78ebaccebc1edb99a4` (ICCV 2025). The original image model, configuration constructors, criteria, alternative backbones, scripts and checkpoint interfaces are retained. The upstream MIT license is in the repository root.

The streaming model now imports the shared `bridgedepth/stereo/` components and `bridgedepth/blocks.py` directly. Its former private `_bridge` copy has been consolidated into those sources.

Shared changes:

- `Backbone` accepts `normalize_input=True` by default, preserving image inputs in [0,255]. Streaming supplies `False` for inputs already in [-1,1]. This adds no parameters or buffers.
- Swin/ConvNeXt torchvision imports occur only when those backbones are constructed. Their architectures and original optional pretrained initialization remain available.
- NMRF's hypothesis FFN input follows its fixed 64+64 concatenation channels and 32 correlation groups. The default 128-dimensional model retains its original parameter shapes; smaller latent dimensions work in regression tests.
- CrossAttention context padding scales with its context/query stride; meshgrid indexing is explicit. This fixes padded half-resolution and double-resolution windows.
- `drop_path` checks `drop_prob` rather than the function name. Image inference and the default zero-dropout configuration retain their behavior.

## FILT3R and CroCo

The bundled architecture and frozen wrapper are unchanged by this integration. FILT3R is pinned to commit `6cf6d28b6ec7fbcc985187a0f099ccbc90224392`. Its provenance and earlier local adaptations are documented in [the bundled README](../bridgedepth/models/_filt3r/README.md). The [FILT3R license](../bridgedepth/models/_filt3r/LICENSE), [CroCo license](../bridgedepth/models/_filt3r/croco/LICENSE) and [CroCo notice](../bridgedepth/models/_filt3r/croco/NOTICE) remain included in the package.

Depth Anything's original [license](../bridgedepth/monocular/LICENSE.txt) is also retained. External checkpoints retain their original terms.

## Streaming refactor

`StreamingBridgeDepth` is organized into streaming orchestration, token adapters and prediction helpers. Learned modules remain registered under their existing names, preserving the previous streaming parameter dictionary format. The frozen FILT3R wrapper and all native inference sources are unchanged. Historical implementation/audit records are kept as written.

Pre-integration random-model regression fixtures cover image inference/training and three-frame streaming with nonzero bridge returns. They validate refactor behavior and checkpoint layout, not pretrained accuracy or runtime performance.
