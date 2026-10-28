"""Stereo prediction helpers; learned heads remain owned by the outer model."""

from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange


def coarse_prediction(
    features: Tensor,
    proposal: Tensor,
    norm: nn.Module,
    head: nn.Module,
    score_head: nn.Module,
) -> tuple[Tensor, Tensor]:
    features = norm(features)
    proposal = proposal.detach().reshape(*features.shape[:4], 1)
    disparity = F.relu(proposal + head(features))
    disparity = rearrange(disparity, "b h w n (hs ws) -> b (h hs) (w ws) n", hs=8)
    logit = 0.25 * score_head(features)
    logit = rearrange(logit, "b h w n (hs ws) -> b (h hs) (w ws) n", hs=8)
    return disparity.contiguous() * 8, logit


def fine_hypothesis(coarse: Tensor, logit: Tensor, stereo: dict, encoder: nn.Module):
    # BridgeDepth's winner selection and lower median, in 1/4-grid units.
    indices = logit.argmax(dim=-1, keepdim=True)
    seed = coarse.gather(-1, indices).squeeze(-1).detach() / 4
    seed = rearrange(seed, "b (h hs) (w ws) -> b h w (hs ws)", hs=4, ws=4)
    seed = seed.sort(dim=-1).values[..., 7:8]
    features, positions = encoder.construct_hypothesis_embedding(
        seed.flatten(0, 2), stereo["fmap1_concat_list"][1],
        stereo["fmap2_concat_list"][1], stereo["fmap1_gw_list"][1],
        stereo["fmap2_gw_list"][1], normalizer=128)
    return seed, features, positions


def fine_prediction(features: Tensor, seed: Tensor, norm: nn.Module, head: nn.Module) -> Tensor:
    update = head(norm(features).squeeze(-2))
    disparity = F.relu(seed + update)
    return rearrange(disparity, "b h w (hs ws) -> b (h hs) (w ws)", hs=4).contiguous() * 4
