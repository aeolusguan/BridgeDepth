"""One-frame FILT3R inference with explicit state and differentiable frozen layers."""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
from torch import Tensor, nn

from ._filt3r.utils.camera import pose_encoding_to_camera


@dataclass(frozen=True)
class FILT3RState:
    state_feat: Tensor
    state_pos: Tensor | None
    mem: Tensor
    state_cov: Tensor | None = None
    prev_candidate_state_feat: Tensor | None = None
    kalman_stats: dict[str, Tensor] | None = None
    frame_index: int = 0

    def detach(self) -> "FILT3RState":
        """Cut the backward graph at a chunk boundary without resetting values."""
        values = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                value = value.detach()
            elif isinstance(value, dict):
                value = {key: tensor.detach() for key, tensor in value.items()}
            values[field.name] = value
        return type(self)(**values)


@dataclass(frozen=True)
class FILT3ROutput:
    depth: Tensor  # [B, 1, H, W], current-camera Z
    camera_pose: Tensor  # [B, 4, 4], OpenCV camera-to-first-camera
    prediction: dict[str, Tensor]  # Native pointmaps, confidence and pose encoding


class FrozenFILT3R(nn.Module):
    """Frozen native FILT3R backbone with explicit streaming state.

    The backbone stays in evaluation mode, while gradients through inputs and
    state remain enabled.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        if not getattr(backbone, "pose_head_flag", False):
            raise ValueError("The native FILT3R backbone must have its pose head enabled.")
        self.backbone = backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "FrozenFILT3R":
        super().train(mode)
        self.backbone.eval()
        return self

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> "FrozenFILT3R":
        """Load a local checkpoint using FILT3R's native loader."""
        from ._filt3r.model import ARCroco3DStereo

        checkpoint = str(Path(checkpoint).expanduser().resolve(strict=True))
        wrapper = cls(ARCroco3DStereo.from_pretrained(checkpoint))
        return wrapper.to(device) if device is not None else wrapper

    def _filter(self, candidate: Tensor, state: FILT3RState) -> tuple[Tensor, Tensor, dict[str, Tensor] | None]:
        # Upstream uses a float32 validity mask, including under autocast.
        update_mask = torch.ones(candidate.shape[0], 1, 1, device=candidate.device, dtype=torch.float32)
        if state.frame_index == 0:
            gain = update_mask
            covariance = torch.full_like(
                candidate[..., 0], self.backbone._get_hparam("kalman_p_init", 1.5)
            )
            statistics = None
        else:
            if state.state_cov is None or state.prev_candidate_state_feat is None:
                raise ValueError("A continuing FILT3R state must include covariance and its previous candidate.")
            gain, covariance, statistics = self.backbone._compute_kalman_ema_gain_and_cov(
                update_mask=update_mask, state_cov=state.state_cov,
                state_feat=state.state_feat, new_state_feat=candidate,
                prev_candidate_state_feat=state.prev_candidate_state_feat,
                cross_attn_state=None, feat_i=None, prev_feat_i=None,
                kalman_stats=dict(state.kalman_stats or {}),
            )
        updated = candidate * gain + state.state_feat * (1 - gain)
        return updated, covariance, statistics

    def step(
        self,
        left: Tensor,
        state: FILT3RState | None = None,
        *,
        true_shape: Tensor | None = None,
    ) -> tuple[FILT3ROutput, FILT3RState]:
        """Process one timestep of a batch of left-view video clips.

        ``left`` has shape [B, 3, H, W], normalized to [-1, 1], with one
        frame from each of B independent clips. Successive calls advance each
        clip by one frame, carrying its state at the same batch index.
        ``state=None`` starts a new batch of clips. Partial resets and
        padded/invalid frames are not supported.
        """
        if left.ndim != 4 or left.shape[1] != 3 or left.shape[0] == 0:
            raise ValueError("left must have shape [B, 3, H, W] with a nonempty batch.")
        batch = left.shape[0]
        if true_shape is None:
            true_shape = torch.tensor(left.shape[-2:], device=left.device).expand(batch, 2)
        if true_shape.shape != (batch, 2):
            raise ValueError("true_shape must have shape [B, 2] in height, width order.")
        true_shape = true_shape.to(left.device)
        features, positions, _ = self.backbone._encode_image(left, true_shape)
        tokens = features[-1]
        if state is None:
            initial, state_positions = self.backbone._init_state(tokens, positions)
            memory = self.backbone.pose_retriever.mem.expand(batch, -1, -1)
            state = FILT3RState(initial, state_positions, memory)
        elif state.state_feat.shape[0] != batch or state.state_feat.device != tokens.device:
            raise ValueError("Continuing state must match the image batch size and device.")
        global_feature = self.backbone._get_img_level_feat(tokens)
        pose = (self.backbone.pose_token.expand(batch, -1, -1) if state.frame_index == 0
                else self.backbone.pose_retriever.inquire(global_feature, state.mem))
        pose_positions = -torch.ones(batch, 1, 2, device=tokens.device, dtype=positions.dtype)
        (candidates, decoded), _ = self.backbone._decoder(
            state.state_feat, state.state_pos, tokens, positions, pose, pose_positions,
            return_attn=False,
        )
        candidate = candidates[-1]
        memory = self.backbone.pose_retriever.update_mem(state.mem, global_feature, decoded[-1][:, :1])
        # Preserve native float32 write-mask promotion under autocast.
        memory = memory.to(torch.promote_types(memory.dtype, torch.float32))
        depth = self.backbone.dec_depth
        head_inputs = [decoded[0].float(), decoded[depth * 2 // 4][:, 1:].float(),
                       decoded[depth * 3 // 4][:, 1:].float(), decoded[depth].float()]
        prediction = self.backbone._downstream_head(head_inputs, true_shape, pos=positions)
        updated, covariance, statistics = self._filter(candidate, state)
        next_state = replace(
            state, state_feat=updated, mem=memory, state_cov=covariance,
            prev_candidate_state_feat=candidate.detach(),
            kalman_stats=statistics, frame_index=state.frame_index + 1,
        )
        output = FILT3ROutput(
            prediction["pts3d_in_self_view"][..., 2].unsqueeze(1),
            pose_encoding_to_camera(prediction["camera_pose"]), prediction,
        )
        return output, next_state

    def forward(
        self,
        left: Tensor,
        state: FILT3RState | None = None,
        *,
        true_shape: Tensor | None = None,
    ) -> tuple[FILT3ROutput, FILT3RState]:
        return self.step(left, state, true_shape=true_shape)
