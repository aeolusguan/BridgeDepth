# Assumption Ledger — shared repository integration
<!-- ASK mode: never -->

| ID | Interface or semantic decision | Chosen | Class | Source |
|----|--------------------------------|--------|-------|--------|
| I-001 | Development branch/base | Original BridgeDepth main on local `dev`; preserve current work; no remote publishing | interface | user |
| I-002 | Model APIs | Original BridgeDepth API preserved; streaming API renamed StreamingBridgeDepth/StreamingBridgeDepthOutput with existing constructor, step and outputs; no V2 aliases | interface | user |
| I-003 | Serialized parameters | Preserve registered module names, parameter/buffer keys and shapes in both models; refactor helpers without nesting registered modules | interface | user |
| I-004 | CNN input convention | Shared Backbone defaults normalize_input=True for RGB [0,255]; streaming passes False for inputs already in [-1,1] | semantic | user |
| I-005 | Shared corrections | Carry over padded cross-resolution attention, drop_path typo correction and fixed hypothesis concatenation width; image defaults retain parameter shapes | semantic | user |
| I-006 | Model computation | Preserve 6+6 interleaving, all gradient detach points, head taps, native fixed-R filter and one state/memory update per frame; FrozenFILT3R and bundled inference source unchanged | semantic | user |
| I-007 | Runtime dependencies | Shared config constructors retain yacs/PyYAML/OmegaConf; torchvision loads only for selected image backbones; FILT3R stays optional | interface | user |
| I-008 | Package and workspace | Distribution bridgedepth, Python namespace bridgedepth; current workspace path retained; historical audit records unchanged | interface | user |
| I-009 | Regression evidence | Before/after comparisons use small random architectures and identical weights, including nonzero bridge returns; establish numerical/interface preservation, not pretrained accuracy | semantic | default |
| I-010 | Installing image inference dependencies | Add optional `image` extra for torchvision, Hugging Face Hub, ImageIO and OpenCV; retain original requirements.txt and installation route for full image training/demo tooling | interface | default |
| I-011 | Inherited dataset ignore rule | Anchor the original dataset-data ignore rule to /datasets/ so the streaming bridgedepth/datasets Python package is tracked and distributed with the Git branch | interface | sweep |

## Notes

- I-005: image numerical equivalence is checked where the upstream implementation runs. Previously failing padded cross-resolution cases are validated against the corrected behavior rather than asserted equivalent to an upstream error.
- I-009: pretrained-weight and compiled-CUDA accuracy/performance validation remains separate from this CPU refactor.
