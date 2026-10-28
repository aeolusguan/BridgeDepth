# 6+6 Bridge Implementation

Target: implement causal stereo video matching with six coarse and six fine BridgeDepth alignment stages interleaved before FILT3R's twelve dual-decoder layers. FILT3R weights stay frozen; stereo/bridges live in an outer model.

Inputs: left/right RGB [B,3,H,W] in [-1,1], H/W divisible by16, optional FILT3RState. Outputs: full-resolution pixel disparity, native monocular depth/pose, next FILT3RState, stereo proposal/coarse/auxiliary predictions for training.

Success command: `/private/tmp/bridgedepthv2-env/bin/python -m pytest -q`.

Base commit: none (not a git repo).

Reuse: aeolusguan/BridgeDepth commit9a231601e6981d3536fa6a78ebaccebc1edb99a4, MIT license. Copy inference primitives only; no external repository/configuration framework needed.

Scope: complete trainable forward/streaming path and gradient verification. Existing loaders remain unchanged. Training runner, task loss policy, previous-view geometric disparity initialization, pretrained accuracy, and CUDA throughput are outside this bridge implementation.

