"""Causal BridgeDepth stereo reasoning interleaved with frozen FILT3R."""

from dataclasses import dataclass, replace
from functools import partial

import torch
from torch import Tensor, nn

from ..stereo.backbone import Backbone
from ..blocks import Alignment
from ..stereo.DPN import DPN
from ..stereo.NMP import Head
from ..stereo.NMRF import NMRF
from ._filt3r.utils.camera import pose_encoding_to_camera
from .filt3r import FILT3RState, FrozenFILT3R
from .bridge import ImageAdapter
from .prediction import coarse_prediction, fine_hypothesis, fine_prediction


@dataclass(frozen=True)
class StreamingBridgeDepthOutput:
    disparity: Tensor  # [B,1,H,W], pixels
    depth: Tensor  # [B,1,H,W], native current-camera Z
    camera_pose: Tensor  # [B,4,4], native camera-to-first-camera
    prediction: dict[str, Tensor]  # Native FILT3R predictions
    stereo: dict  # BridgeDepth disparity/proposal/auxiliary predictions


class StreamingBridgeDepth(nn.Module):
    """Six 1/8 and six 1/4 stereo stages, each followed by one dual layer.

    FILT3R image tokens keep their native 1/16 grid. Monocular alignment
    features use 1/8 at both stages, following BridgeDepth's fine-stage
    cross-resolution attention. Only FILT3RState is carried between frames.
    """

    def __init__(
        self,
        monocular: FrozenFILT3R,
        *,
        embed_dim: int = 128,
        stereo_feature_dim: int = 256,
        max_disparity: int = 320,
        num_proposals: int = 2,
        num_heads: int = 4,
        proposal_layers: int = 5,
        window_size: int = 6,
        refine_window_size: int = 4,
    ):
        super().__init__()
        backbone = monocular.backbone
        if backbone.dec_depth != 12:
            raise ValueError("The 6+6 bridge requires twelve FILT3R dual-decoder layers.")
        if tuple(backbone.patch_embed.patch_size) != (16, 16):
            raise ValueError("The bridge requires FILT3R's 16x16 image patches.")
        if max_disparity % 8 or max_disparity < 8 * num_proposals:
            raise ValueError("max_disparity must be divisible by8 and cover num_proposals.")
        if refine_window_size % 2:
            raise ValueError("refine_window_size must be even for 1/4-to-1/8 attention.")

        self.monocular = monocular
        dpn = DPN(cost_group=4, num_proposals=num_proposals,
                  feat_dim=stereo_feature_dim, prior_dim=embed_dim,
                  num_prop_layers=proposal_layers, prop_embed_dim=embed_dim,
                  mlp_ratio=4, split_size=1, prop_n_heads=num_heads)
        self.stereo_encoder = NMRF(Backbone(stereo_feature_dim, normalize_input=False), dpn,
                                   num_proposals, max_disparity, embed_dim)
        alignment_args = dict(num_blocks=6, embed_dim=embed_dim, num_heads=num_heads,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.align_0 = Alignment(window_size=window_size, refine=False, **alignment_args)
        self.align_1 = Alignment(window_size=refine_window_size, refine=True, **alignment_args)
        self.adapters = nn.ModuleList(
            ImageAdapter(backbone.dec_embed_dim, embed_dim) for _ in range(12))
        self.infer_head = Head(embed_dim, embed_dim, 8 * 8, num_layers=3)
        self.infer_score_head = nn.Linear(embed_dim, 8 * 8)
        self.refine_head = Head(embed_dim, embed_dim, 4 * 4, num_layers=3)

    def _coarse_prediction(self, features: Tensor, proposal: Tensor) -> tuple[Tensor, Tensor]:
        return coarse_prediction(features, proposal, self.align_0.norm,
                                 self.infer_head, self.infer_score_head)

    def _fine_hypothesis(self, coarse: Tensor, logit: Tensor, stereo: dict):
        return fine_hypothesis(coarse, logit, stereo, self.stereo_encoder)

    def _fine_prediction(self, features: Tensor, seed: Tensor) -> Tensor:
        return fine_prediction(features, seed, self.align_1.norm, self.refine_head)

    def step(
        self,
        left: Tensor,
        right: Tensor,
        state: FILT3RState | None = None,
        *,
        true_shape: Tensor | None = None,
    ) -> tuple[StreamingBridgeDepthOutput, FILT3RState]:
        """Process one stereo timestep, RGB [B,3,H,W] normalized to [-1,1].

        Each batch index belongs to one independent clip. Keep that order on
        subsequent calls; state=None starts new clips. H and W must be
        divisible by16. No future frames, padded timesteps or partial resets.
        """
        if left.ndim != 4 or left.shape[1] != 3 or left.shape[0] == 0 or right.shape != left.shape:
            raise ValueError("left and right must have matching nonempty [B,3,H,W] shapes.")
        batch, _, height, width = left.shape
        if height % 16 or width % 16:
            raise ValueError("Stereo image height and width must be divisible by16.")
        if true_shape is None:
            true_shape = torch.tensor((height, width), device=left.device).expand(batch, 2)
        if true_shape.shape != (batch, 2):
            raise ValueError("true_shape must have shape [B,2] in height,width order.")
        true_shape = true_shape.to(left.device)
        backbone = self.monocular.backbone
        encoded, positions, _ = backbone._encode_image(left, true_shape)
        tokens = encoded[-1]
        if state is None:
            initial, state_positions = backbone._init_state(tokens, positions)
            state = FILT3RState(initial, state_positions,
                                backbone.pose_retriever.mem.expand(batch, -1, -1))
        elif state.state_feat.shape[0] != batch or state.state_feat.device != tokens.device:
            raise ValueError("Continuing state must match the image batch size and device.")

        global_feature = backbone._get_img_level_feat(tokens)
        pose = (backbone.pose_token.expand(batch, -1, -1) if state.frame_index == 0
                else backbone.pose_retriever.inquire(global_feature, state.mem))
        pose_positions = -torch.ones(batch, 1, 2, device=tokens.device, dtype=positions.dtype)
        image_positions = torch.cat((pose_positions, positions), dim=1)
        image = torch.cat((pose, backbone.decoder_embed(tokens)), dim=1)
        patch_shape = (height // 16, width // 16)
        prior = self.adapters[0].read(image[:, 1:], patch_shape)
        stereo = self.stereo_encoder({"img1": left, "img2": right, "depth_prior": prior})
        features, stereo_positions = self.stereo_encoder.construct_hypothesis_embedding(
            stereo["proposal"].flatten(0, 1).detach(), stereo["fmap1_concat_list"][0],
            stereo["fmap2_concat_list"][0], stereo["fmap1_gw_list"][0],
            stereo["fmap2_gw_list"][0])
        masks = self.align_0.compute_attn_mask(*features.shape[1:4], features.device)

        scene = state.state_feat
        decoded = [tokens]  # Native encoder skip for DPT.
        coarse_outputs, fine_outputs = [], []
        for index, (state_block, image_block, adapter) in enumerate(zip(
                backbone.dec_blocks_state, backbone.dec_blocks, self.adapters)):
            if index == 6:
                seed, features, stereo_positions = self._fine_hypothesis(
                    coarse_outputs[-1][0], coarse_outputs[-1][1], stereo)
                masks = self.align_1.compute_attn_mask(*features.shape[1:4], features.device)
            alignment = self.align_0 if index < 6 else self.align_1
            stage_index = index % 6
            mono = prior if index == 0 else adapter.read(image[:, 1:], patch_shape)
            features, refined_mono = alignment.layers[stage_index](
                features, mono.unsqueeze(-2), stereo_positions, masks[stage_index % 2])
            correction = adapter.write(refined_mono.squeeze(-2) - mono, patch_shape)
            image = torch.cat((image[:, :1], image[:, 1:] + correction), dim=1)

            # Both native branches consume this same previous-layer snapshot.
            candidate = state_block(scene, image, state.state_pos, image_positions,
                                    return_attn=False)[0]
            image = image_block(image, scene, image_positions, state.state_pos,
                                return_attn=False)[0]
            scene = candidate
            decoded.append(image)
            if index < 6 and (self.training or index == 5):
                coarse_outputs.append(self._coarse_prediction(features, stereo["proposal"]))
            elif index >= 6 and (self.training or index == 11):
                fine_outputs.append(self._fine_prediction(features, seed))

        candidate = backbone.dec_norm_state(scene)
        decoded[-1] = backbone.dec_norm(decoded[-1])
        prediction = backbone._downstream_head(
            [decoded[0].float(), decoded[6][:, 1:].float(),
             decoded[9][:, 1:].float(), decoded[12].float()], true_shape, pos=positions)
        memory = backbone.pose_retriever.update_mem(state.mem, global_feature, decoded[-1][:, :1])
        memory = memory.to(torch.promote_types(memory.dtype, torch.float32))
        updated, covariance, statistics = self.monocular._filter(candidate, state)
        next_state = replace(state, state_feat=updated, mem=memory, state_cov=covariance,
                             prev_candidate_state_feat=candidate.detach(),
                             kalman_stats=statistics, frame_index=state.frame_index + 1)

        stereo_prediction = dict(disp_pred=fine_outputs[-1], coarse_disp=coarse_outputs[-1][0],
                                 logit=coarse_outputs[-1][1], proposal=stereo["proposal"],
                                 prob=stereo["prob"])
        if self.training:
            stereo_prediction["aux_outputs"] = [
                dict(disp_pred=disparity, aux=i) for i, disparity in enumerate(fine_outputs[:-1])]
            stereo_prediction["aux_outputs"] += [
                dict(coarse_disp=disparity, logit=logit, aux=i)
                for i, (disparity, logit) in enumerate(coarse_outputs[:-1])]
        output = StreamingBridgeDepthOutput(
            disparity=fine_outputs[-1].unsqueeze(1),
            depth=prediction["pts3d_in_self_view"][..., 2].unsqueeze(1),
            camera_pose=pose_encoding_to_camera(prediction["camera_pose"]),
            prediction=prediction, stereo=stereo_prediction)
        return output, next_state

    def forward(self, left: Tensor, right: Tensor, state: FILT3RState | None = None,
                *, true_shape: Tensor | None = None) -> tuple[StreamingBridgeDepthOutput, FILT3RState]:
        return self.step(left, right, state, true_shape=true_shape)
