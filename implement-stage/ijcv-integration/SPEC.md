# Shared BridgeDepth repository integration

Target: integrate the existing causal stereo implementation into aeolusguan/BridgeDepth on local branch `dev`, preserving the image model and adding `StreamingBridgeDepth`.

Inputs: the current workspace, original repository main (expected `9a231601e6981d3536fa6a78ebaccebc1edb99a4`), and saved pre-refactor streaming outputs/weights.

Outputs: shared stereo/alignment implementation, separate streaming orchestration/adapters/prediction helpers, unchanged FILT3R sources, preserved image interfaces/checkpoints, updated package metadata and documentation, and regression tests.

Success command: `python -m pytest -q`, plus numerical before/after comparisons and wheel-content inspection.

Starting base: none (workspace initially has no Git repository). Integration base: `9a231601e6981d3536fa6a78ebaccebc1edb99a4`, verified before attaching Git history on branch `dev`. Current source is preserved at `/private/tmp/bridgedepth-ijcv-before` before integration.

Scope cuts: no remote push, training runner, new loss, next-frame disparity warping, or filesystem/project-folder rename. Historical implementation records remain unchanged.
