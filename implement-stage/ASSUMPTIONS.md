# Assumption Ledger — 6+6 Bridge
<!-- ASK mode: never -->

| ID | Under-determined by request | Chosen | Class | Source |
|---|---|---|---|---|
| A-001 | Exact per-layer placement | Alignment before each of12 dual layers; first6 stereo1/8, last6 stereo1/4; preserve synchronous image/state inputs | semantic | user |
| A-002 | Cross-resolution interface | Preserve BridgeDepth monocular alignment grid1/8 at both stages; project/resize current FILT3R1/16 image tokens into it and return only the alignment residual to native grid | semantic | default |
| A-003 | Persistent monocular bridge feature | Recompute from current decoder tokens each layer so temporal context enters every exchange; no separate carried high-resolution latent | semantic | default |
| A-004 | Initialization and native heads | Zero-initialize only stereo-to-FILT residual output projections; preserve raw encoder DPT skip, pose token, positions, native head taps, raw encoder pose-memory keys; filter/memory commit once perframe | semantic | default |
| A-005 | Stereo components/defaults | Copy BridgeDepth MIT ResNet backbone,DPN,NMP,alignment and disparity head math; native defaults128latent,256CNN,2proposals,maxdisp320; six layers replace each original five | semantic | default |
| A-006 | Input/output interface | Outer BridgeDepthV2(left,right,state=None,true_shape=None) returns(output,state); normalized[-1,1] inputs, aligned fixed resolution divisible by16; disparity pixels [B,1,H,W], depth/pose native outputs | interface | default |
| A-007 | Stereo initialization across video | Each frame's DPN uses projected current encoder features; state carries temporal context through FILT3R; explicit pose/depth warping to next-frame proposal initialization is separate work | semantic | default |
| A-008 | External source dependency | Vendor selected source from BridgeDepth commit9a231601e6981d3536fa6a78ebaccebc1edb99a4 with MIT license/provenance; remove configuration decorators and unused alternate backbones/training criteria | interface | default |
| A-009 | Checkpoint compatibility | Load FILT3R through existing native loader; new outer state_dict uses standard PyTorch. Existing BridgeDepth full weights are not directly shape-compatible with6+6 and new adapters | interface | default |
| A-010 | Training outputs/supervision | Expose native stereo coarse/logit/proposal/prob and per-layer auxiliary disparities; no pose loss or required pose labels introduced | semantic | user |
| A-011 | DPN coordinate gradients during stereo refinement | Detach proposal coordinates in hypothesis sampling, coarse regression and coarse-to-fine selection, matching BridgeDepth; DPN learns from explicit proposal/probability losses on returned tensors | semantic | sweep |

## Notes

A-002 reuses BridgeDepth's fine-stage1/4↔1/8 attention rather than introducing a new1/4 monocular bridge representation. Returning a learned correction to1/16 preserves FILT3R token/position interfaces.

