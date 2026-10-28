# Streaming stereo

The streaming extension is in development for the journal extension of BridgeDepth. It targets calibrated, causal stereo video. There is one stereo dataset loader; the frozen FILT3R branch takes its left frames and needs no separate monocular dataset:

- `bridgedepth/datasets/`: Scene Flow and Dynamic Replica clips, optional poses, shared geometric augmentation.
- `bridgedepth/models/filt3r.py`: a frozen FILT3R backbone that consumes one left image at a time and returns an explicit recurrent state.
- `bridgedepth/models/streaming.py`: streaming orchestration and six coarse plus six fine alignment stages interleaved with the twelve dual-decoder layers.
- `bridgedepth/models/bridge.py`: adapters between native FILT3R tokens and alignment features.
- `bridgedepth/models/prediction.py`: stereo prediction helpers; learned heads remain registered on the streaming model.
- `bridgedepth/stereo/` and `bridgedepth/blocks.py`: shared image/video stereo and bidirectional alignment components.
- `bridgedepth/models/_filt3r/`: the bundled FILT3R inference architecture and required CroCo components.
- `tests/`: native-format data fixtures, geometry checks, freezing and temporal-gradient checks.

The streaming stereo forward path is implemented. Task losses and a full training runner remain separate work.

## Install

Run these commands from the repository root. Image-model instructions remain in the [main README](../README.md).

```bash
python -m pip install -e '.[test,filt3r]'
```

The required FILT3R inference source is included in this package. No FILT3R, CUT3R or CroCo checkout is needed. The model uses PyTorch, NumPy, Transformers and Einops; Pillow is used by the shared stereo loader. Demo, visualization and upstream dataset dependencies are excluded.

Download `cut3r_512_dpt_4_64.pth` using the [official checkpoint link](https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link) and place it in `checkpoints/`. Model loading only needs this file path.

For faster positional encoding on the CUDA training machine, optionally build the included kernel:

```bash
(cd bridgedepth/models/_filt3r/croco/curope && python setup.py build_ext --inplace)
```

Without the extension, the bundled PyTorch implementation runs on CPU or CUDA and supports the pose token's negative position. Building the extension requires a CUDA toolkit compatible with PyTorch. Source provenance, local changes and original licenses are kept in [bundled FILT3R sources](../bridgedepth/models/_filt3r/).

## Dataset layouts

Scene Flow uses the **full release**, not the flattened FlyingThings3D subset:

```text
SceneFlow/
  FlyingThings3D/
    frames_finalpass/TRAIN/A/0000/{left,right}/0006.png
    disparity/TRAIN/A/0000/left/0006.pfm
    camera_data/TRAIN/A/0000/camera_data.txt     # optional
  Monkaa/
    frames_finalpass/<sequence>/{left,right}/0000.png
    disparity/<sequence>/left/0000.pfm
  Driving/
    frames_finalpass/15mm_focallength/scene_forwards/fast/{left,right}/0001.png
    disparity/15mm_focallength/scene_forwards/fast/left/0001.pfm
```

`render_pass="clean"` selects `frames_cleanpass`. Supply only the subsets you have. Only FlyingThings3D has an official `test` split. The loader expects original 960×540 images; resize with the supplied transform so calibration stays consistent. Camera files are optional, and disparity/depth remain usable without them. Scene Flow depth, baseline and translation use **Blender units**, not metres (`metric_scale=False`).

Dynamic Replica uses the released annotations and paths:

```text
DynamicReplica/
  train/
    frame_annotations_train.jgz
    <image and depth paths listed in the annotations>
  valid/
    frame_annotations_valid.jgz
  test/
    frame_annotations_test.jgz
```

The dataset joins left/right annotations by sequence and frame number. It decodes depth PNG values as float16 bit patterns and converts PyTorch3D camera conventions to OpenCV. Depth units are metres. If R/T annotations are missing, depth still loads; a known `baseline=...` in metres additionally enables disparity conversion. Intrinsics are required for this calibrated setup. `load_poses=False` suppresses pose targets while retaining available rig calibration.

## Clip batching

```python
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader
from bridgedepth.datasets import SceneFlowClips, DynamicReplicaClips, StereoClipTransform

np.random.seed(0)  # reproduce the Dynamic Replica clip list
transform = StereoClipTransform(resize=(288, 512), crop_size=(256, 512))
sf = SceneFlowClips(
    "/data/SceneFlow", subsets=("FlyingThings3D",), split="train",
    render_pass="final", clip_length=5, transform=transform,
)
dr = DynamicReplicaClips(
    "/data/DynamicReplica", split="train", clip_length=5, transform=transform,
)
loader = DataLoader(
    ConcatDataset([sf, dr]), batch_size=1, shuffle=True,
    drop_last=True, num_workers=4,
)
batch = next(iter(loader))
```

`batch_size` counts **independent clips**, each with `clip_length` ordered stereo pairs. Training clip sampling follows the [DynamicStereo](https://github.com/facebookresearch/dynamic_stereo/blob/main/datasets/dynamic_stereo_datasets.py) and [Stereo Any Video](https://github.com/TomTomTommi/stereoanyvideo/blob/main/datasets/video_datasets.py) loaders:

- **Scene Flow:** consecutive forward and reversed clips at every candidate start.
- **Dynamic Replica:** candidate starts every three frames, with a stride sampled uniformly from 1–5 at dataset construction. Single-frame clips use stride 1. The sampled clip list stays fixed across epochs; seed NumPy before constructing the datasets for reproducibility.

The default clip length is five. Training retains the reference loaders' end-boundary rules; a sequence of exactly `clip_length` frames produces no training clip. Validation/test splits use chronological, non-overlapping complete clips and omit incomplete tails. Clips never cross a scene boundary or a missing frame.

Reversal applies to both images and all labels, intrinsics, poses, and frame IDs together. The model remains causal in the supplied clip order; reset recurrent state for each new clip. `ConcatDataset` samples in proportion to clip counts. The old fixed `frame_stride` and `clip_stride` options are replaced by these dataset-specific policies.

| Key | Batched shape | Meaning |
| --- | --- | --- |
| `images` | B,T,2,3,H,W | RGB float32 in [0,1]; left=0, right=1 |
| `disparity`, `depth` | B,T,1,H,W | Left disparity in pixels; camera-axis Z depth |
| `disparity_valid`, `depth_valid` | B,T,1,H,W | Boolean supervision masks |
| `intrinsics` | B,T,2,3,3 | Pixel intrinsics after spatial transforms |
| `baseline`, `baseline_valid` | B,T | Baseline in depth units and availability |
| `poses`, `pose_valid` | B,T,4,4 and B,T | OpenCV left camera-to-world and availability |
| `frame_ids` | B,T | Original frame numbers |
| `metric_scale` | B | Whether depth/translation units are metres |
| `scene_id`, `dataset`, `depth_unit` | B strings | Source metadata |

Missing poses use identity placeholders with `pose_valid=False`. They are **not identity-pose supervision**. Missing/invalid dense labels are zero with false masks. Pose loss must be omitted for missing labels. For pose loss in the first-camera frame, use `inverse(C0) @ Ct`, only where both poses are valid, and keep that anchor across chunks. Use scale-aware supervision for Scene Flow rather than treating its depth as metres.

Resize/crop parameters are shared over all times and both cameras. Intrinsics use the same pixel-center convention as the image resize; disparity scales with width, while depth and translation do not scale. The conversion is `d = fx * baseline / Z + cx_left - cx_right`. There is no right-view vertical jitter.

## Streaming left-view model

```python
import torch
from bridgedepth.models import FrozenFILT3R

model = FrozenFILT3R.from_pretrained(
    "checkpoints/cut3r_512_dpt_4_64.pth",
    device="cuda",
)

state = None  # reset once for each new batch of unrelated clips
with torch.no_grad():  # pure inference
    for t in range(batch["images"].shape[1]):
        left = batch["images"][:, t, 0].cuda() * 2 - 1
        output, state = model(left, state)
        depth = output.depth             # B,1,H,W: pts3d_in_self_view[...,2]
        pose = output.camera_pose        # B,4,4: camera-to-first-camera
        native = output.prediction       # pointmaps, confidence, native pose encoding
```

Use image dimensions divisible by the native patch size (16 for the supplied checkpoint). The wrapper uses FILT3R's native fixed-R filter, irrespective of the checkpoint's `model_update_type` default. It carries scene tokens, pose memory, covariance, candidate history and drift EMA. `state=None` starts a new clip; `state.detach()` preserves values while removing their autograd history. Keep batch membership/order fixed within a stream. Partial batch resets and padded timesteps are deliberately unsupported.

For comparison with the official demo, select `--model_update_type filt3r`; its CLI defaults to `cut3r`. The bundled configuration matches the demo's FILT3R filter defaults. The demo uses float32 tensors with autocast disabled and enables CUDA TF32; the wrapper leaves runtime precision settings to the caller. The demo prepares RGB images by resizing the long edge to 512 and center cropping to patch-compatible dimensions. This wrapper accepts already-prepared tensors; use identical resize/crop, normalization and precision settings for numerical comparisons. Stereo training keeps its shared calibrated transforms.

`FrozenFILT3R` owns only the frozen monocular backbone and its streaming transition. Add stereo and bridging modules in your outer wrapper; `model.backbone` exposes the native FILT3R components. All FILT3R parameters have `requires_grad=False`, and the backbone stays in evaluation mode even when the outer model enters training mode.

Frozen FILT3R inference uses `torch.no_grad()` and needs neither TBPTT nor explicit state detachment. If trainable components feed features or state into FILT3R, leave gradients enabled through the frozen layers. You can backpropagate through the whole training clip when memory permits. For longer unrolls, optionally detach state at chunk boundaries to limit gradient history and memory (TBPTT). Reset state when starting a different shuffled clip.

## Streaming stereo bridge

```python
from bridgedepth.models import StreamingBridgeDepth, FrozenFILT3R

monocular = FrozenFILT3R.from_pretrained("checkpoints/cut3r_512_dpt_4_64.pth")
model = StreamingBridgeDepth(monocular).cuda().eval()
state = None
with torch.no_grad():
    for t in range(batch["images"].shape[1]):
        pair = batch["images"][:, t].cuda() * 2 - 1
        output, state = model(pair[:, 0], pair[:, 1], state)
        disparity = output.disparity   # B,1,H,W; pixels
        depth = output.depth           # B,1,H,W; native FILT3R camera-axis Z
        pose = output.camera_pose      # B,4,4; camera-to-first-camera
```

Both inputs are RGB `[B,3,H,W]` in `[-1,1]`, with matching dimensions divisible by16. The supplied FILT3R architecture must use sixteen-pixel patches and twelve decoder layers. State has the same meaning and reset/detach behavior as `FrozenFILT3R`; the stereo branch consumes only the current pair.

Each stage performs BridgeDepth's monocular readout, NMRF message passing and monocular update, then executes the corresponding frozen dual-decoder layer. Layers1–6 use stereo hypotheses at1/8 resolution; layers7–12 use a single refined hypothesis at1/4. The transition retains BridgeDepth's coarse winner selection and lower-median initialization, with detached disparity coordinates.

Monocular bridge features stay at1/8 at both stages, as in BridgeDepth. At each layer a separate adapter projects the **current** FILT3R image tokens from1/16 to1/8. Alignment returns a residual that is projected and resized back to1/16. Only that residual is added to the native image tokens. The pose token and positional coordinates retain their native interfaces. The encoder DPT skip is unchanged; DPT still reads features at0,6,9,12. Both halves of a dual layer read the same previous-layer scene/image pair. The scene filter and pose-memory write run once per frame.

All stereo-to-FILT3R output projections start at zero. Initial monocular predictions and recurrent state therefore reproduce the standalone frozen model. Those projections can learn immediately; gradients reach earlier stereo-to-monocular update parameters once the return projections become nonzero. FILT3R remains frozen and in evaluation mode when `model.train()` is called, while the stereo and bridge modules train normally.

For training, call `model.train()` and leave gradients enabled. Optimize parameters with `requires_grad=True`, accumulate losses over the ordered clip, and reset state for a new shuffled clip. Pose labels are optional; this forward path introduces no required pose loss. The return dictionary `output.stereo` retains BridgeDepth's training interface:

| Key | Shape / units |
|---|---|
| `disp_pred` | B,H,W; pixels, also available as `output.disparity[:,0]` |
| `coarse_disp`, `logit` | B,H,W,N; coarse pixel disparities and hypothesis scores |
| `proposal` | B,(H/8 W/8),N; disparity in1/8-grid units; multiply by8 for pixels |
| `prob` | (B H/8 W/8),D; D=`max_disparity/8` initial probabilities |
| `aux_outputs` | Training only: five fine disparity and five coarse disparity/score dictionaries |

Proposal coordinates are detached for hypothesis sampling, coarse regression and stage transition, matching BridgeDepth. Train DPN with explicit proposal/probability losses using the returned `proposal` and `prob`; the final disparity loss alone does not train it.

Defaults follow BridgeDepth's CNN stereo configuration:128 latent channels,256 CNN output channels, two proposals, maximum disparity320, five DPN propagation layers, coarse windows6 and fine windows4. Only the alignment stage counts change from5+5 to6+6. `embed_dim`, `stereo_feature_dim`, `max_disparity`, `num_proposals`, `num_heads`, `proposal_layers`, `window_size`, and `refine_window_size` are constructor arguments.

Depth is decoded by FILT3R; it is not forcibly converted from stereo disparity. Each frame starts stereo proposals with DPN conditioned on current encoder features; temporal context enters later alignment through the scene-conditioned decoder features. Explicit pose/depth warping to initialize the next frame's disparity is not implemented here.

Load FILT3R through its existing loader, and save/load the outer model with standard PyTorch `state_dict` operations. A complete original BridgeDepth checkpoint is not directly compatible with the new adapters and6+6 stack. Stereo and alignment components are shared with the original image model in this repository. Shared constructors retain the original configuration support; the streaming model uses explicit constructor arguments. Its CNN needs no torchvision import or automatic backbone download. Source provenance and shared changes are listed in [source provenance](source_provenance.md).

## Checks and references

```bash
python -m pytest -q
# Bundled architecture, streaming recurrence, checkpoint loading and gradients:
python -m pytest -q tests/test_filt3r_integration.py tests/test_vendored_filt3r.py
# Stereo primitives,6+6 scheduling, native equivalence and temporal gradients:
python -m pytest -q tests/test_bridge.py tests/test_repository_integration.py
```

These tests run entirely from this project, without an external checkout, pretrained weights or a GPU. They use small randomly initialized linear/DPT architectures and the actual bundled positional encoding. They compare streaming outputs against the demo's `forward_recurrent_lighter` path over five frames with batch size two, including filter gains, covariance and drift EMA. They also cover recurrent state, pose memory, freezing and gradients; pretrained accuracy and compiled CUDA performance require separate GPU validation.

Bridge checks use the actual CNN, DPN, NMRF and twelve-layer native decoder. They cover cross-resolution padding,6+6 ordering, the shared dual-layer input snapshot, zero-residual native equivalence, disparity units/selection, training auxiliaries, optimizer gradients with frozen FILT3R, and last-bridge influence on current outputs and future scene/stereo predictions. Random tensors in these checks establish execution/gradient behavior only.

Validated with Python 3.10, PyTorch 2.6.0, Transformers 4.49.0 and FILT3R commit `6cf6d28b6ec7fbcc985187a0f099ccbc90224392`.

The implementation follows the clip-based data organization of [DynamicStereo](https://github.com/facebookresearch/dynamic_stereo/blob/main/datasets/dynamic_stereo_datasets.py) and [Stereo Any Video](https://github.com/TomTomTommi/stereoanyvideo/blob/main/datasets/video_datasets.py), with causal ordering and calibrated spatial transforms. The recurrent transition follows [FILT3R](https://github.com/jinotter3/FILT3R/blob/main/src/dust3r/model.py). Dataset conventions are documented by [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) and [Dynamic Replica](https://github.com/facebookresearch/dynamic_stereo). External code and checkpoints retain their original licenses.
