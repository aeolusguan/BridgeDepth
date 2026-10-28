# Build Note — 6+6 Bridge

| Rung | Feature | Acceptance check | Tier | Status |
|---|---|---|---|---|
| F0 | Existing native streaming spine | `python -m pytest -q tests/test_filt3r_integration.py` | MUST | passed:6tests |
| F1 | Vendored stereo and alignment primitives | `python -m pytest -q tests/test_bridge.py -k primitives` | MUST | passed:4tests |
| F2 | Interleaved12-layer model and native no-op equivalence | `python -m pytest -q tests/test_bridge.py -k 'schedule or native'` | MUST | passed:3tests |
| F3 | Stereo/current-output/recurrent gradients and training outputs | `python -m pytest -q tests/test_bridge.py` | MUST | passed:13bridge tests |
| F4 | Usage documentation and accumulated checks | `python -m pytest -q` | MUST | passed:50tests |

## Run record

Reference source retrieved through official GitHub connector. No local GPU or pretrained checkpoint assumed. Base working directory has no Git repository; no commits will be created.

F0: temporary Python3.10 environment with Torch2.6, Transformers4.49; native streaming checks exited0 (6passed). Initial direct PyPI download was slow; installation completed using the Tsinghua mirror.

F1: four primitive checks exited0: padded1x/0.5x/2x attention forward/backward and real CNN/DPN/coarse alignment backward. Copied components do not depend on torchvision or the upstream configuration framework.

F2: bridge+existing native integration suite exited0 (13passed). Linear/DPT predictions and recurrent values match the original wrapper over3frames, B2; hooks verify A_i→Dstate_i→Dimage_i, same input snapshot,6coarse/6fine, one filter and pose-memory write perframe.

F3: bridge+native integration exited0 (18passed). Depth/disparity-only optimizer step updates the bridge while native weights stay frozen. Opening only the last bridge changes current depth/pose and scene state; next-frame mono loss reaches the prior right image through scene state with pose memory detached. Future stereo loss has nonzero scene-state gradients. Real default128/256-channel inference and non-window-multiple48x80 input pass. Coarse/fine pixel units, winning hypothesis and lower median are checked.

F4: full suite exited0 (49passed,12existing native AMP deprecation warnings). The sandboxed run passed48 tests but its existing subprocess import check hit OpenMP shared-memory restrictions; the authorized unsandboxed rerun passed all49. README and package data include the new model, vendored license/provenance and scope limits.

Review route: reviewer_route.py could not be resolved through ARIS_REPO/manifest, projecttools or skill directory; no project reviewer configuration exists. Using the documented Codex host default, fresh gpt-6-astra/xhigh. Same-family review is provisional; it does not validate model accuracy or research claims.

Sweep round1 found one semantic mismatch: coarse regression used live DPN proposal coordinates. A-011 was added before correcting the code to match upstream detachment; a regression test now checks that boundary. The complete suite after this fix exited0:50passed in10.26s. Round2 found zero remaining undeclared semantic/interface choices and no stale ledger rows. Reviewer identity was returned as unknown; requested model/effort and both raw JSON responses are preserved in SILENT_ASSUMPTION_SWEEP.json. Status remains same-family/provisional.

Additional verification: two-frame CPU bfloat16 autocast preserves zero-bridge native depth/state and float32 recurrent writes; wheel builds and includes the bridge sources and both upstream licenses. AST comparison of copied NMP definitions shows only the documented drop_path typo fix; removed unused containers are documented in provenance.

All MUST rungs pass; no live stubs. Semantic and interface decisions are recorded in full in ASSUMPTIONS.md. No pretrained accuracy, long-video stability or CUDA throughput claim is made.

## Deferred

Training runner and loss policy; previous-view pose/depth geometric proposal initialization; CUDA performance and pretrained evaluation.

## Blockers

None.

