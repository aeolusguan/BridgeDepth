from typing import Union, IO
from pathlib import Path
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import hf_hub_download

from .config import CfgNode
from .monocular.depth_anything import DepthAnything
from .stereo import build_model as build_stereo_encoder
from .utils.frame_utils import InputPadder
from .blocks import Alignment
from .stereo.NMP import Head
    

class BridgeDepth(nn.Module):

    def __init__(
        self,
        cfg,
        mono_pretrained=False,
    ):
        super().__init__()

        encoder = cfg.BRIDGEDEPTH.ENCODER_VIT
        self.encoder = encoder
        self.padder = None

        # monocular depth branch
        if mono_pretrained:
            pretrained_model_name_or_path = f'checkpoints/depth_anything_v2_{encoder}.pth'
            if not os.path.exists(pretrained_model_name_or_path):
                repo_type = {'vitl': 'Large', 'vitb': 'Base', 'vits': 'Small'}
                pretrained_model_name_or_path = f"depth-anything/Depth-Anything-V2-{repo_type[encoder]}"
            self.mono = DepthAnything.from_pretrained(pretrained_model_name_or_path)
        else:
            self.mono = DepthAnything(encoder)
        self.mono.freeze()
        depth_feature_dim = self.mono.feature_dim
        dim = cfg.NMP.INFER_EMBED_DIM
        self.depth_feature_down = nn.Conv2d(depth_feature_dim, dim, kernel_size=4, stride=4, padding=0)
        
        # stereo branch
        self.stereo_encoder = build_stereo_encoder(cfg)
        
        # Alignment
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.align_0 = Alignment(num_blocks=cfg.NMP.NUM_INFER_LAYERS, embed_dim=dim, window_size=cfg.NMP.WINDOW_SIZE, 
                                 num_heads=cfg.NMP.INFER_N_HEADS, mlp_ratio=cfg.NMP.MLP_RATIO, drop=cfg.NMP.DROPOUT, 
                                 attn_drop=cfg.NMP.ATTN_DROP, drop_path=cfg.NMP.DROP_PATH, norm_layer=norm_layer, 
                                 return_intermediate=cfg.NMP.RETURN_INTERMEDIATE, refine=False)
        self.align_1 = Alignment(num_blocks=cfg.NMP.NUM_REFINE_LAYERS, embed_dim=dim, window_size=cfg.NMP.REFINE_WINDOW_SIZE, 
                                 num_heads=cfg.NMP.INFER_N_HEADS, mlp_ratio=cfg.NMP.MLP_RATIO, drop=cfg.NMP.DROPOUT, 
                                 attn_drop=cfg.NMP.ATTN_DROP, drop_path=cfg.NMP.DROP_PATH, norm_layer=norm_layer, 
                                 return_intermediate=cfg.NMP.RETURN_INTERMEDIATE, refine=True)
        
        # output head
        self.infer_head = Head(dim, dim, 8*8, num_layers=3)
        self.infer_score_head = nn.Linear(dim, 8*8)
        self.refine_head = Head(dim, dim, 4*4, 3)
        self.mono_output = Head(dim, dim, 8*8, num_layers=3)

        # mean and std of the pretrained dinov2 model
        self.register_buffer(
            "mean", torch.as_tensor([123.675, 116.28, 103.53], dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.as_tensor([58.395, 57.12, 57.375], dtype=torch.float32)[None, :, None, None]
        )

    @property
    def device(self):
        return self.stereo_encoder.device
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path, IO[bytes]], **hf_kwargs) -> 'BridgeDepth':
        """
        Load a model from a checkpoint file.

        Args:
            pretrained_model_name_or_path: path to the checkpoint file or repo id.
            hf_kwargs: additional keyword arguements to pass to the hf_hub_download function. Ignored if pretrained_model_name_or_path is a local path.

        Returns:
            a new instance of `BridgeDepth` with the parameters loaded from the checkpoint.
        """
        torch.serialization.add_safe_globals([CfgNode])
        if Path(pretrained_model_name_or_path).exists():
            checkpoint = torch.load(pretrained_model_name_or_path, map_location='cpu', weights_only=True)
        else:
            cached_checkpoint_path = hf_hub_download(
                repo_id="aeolusguan/BridgeDepth",
                repo_type="model",
                filename=f"bridge_{pretrained_model_name_or_path}.pth",
                **hf_kwargs,
            )
            checkpoint = torch.load(cached_checkpoint_path, map_location='cpu', weights_only=True)
        model_config = checkpoint['model_config']
        model = cls(model_config)
        model.load_state_dict(checkpoint['model'])
        return model
    
    def prepare_input(self, inputs):
        img1 = inputs["img1"].to(self.device)
        img2 = inputs["img2"].to(self.device)

        if not self.training:
            self.padder = InputPadder(img1.shape, mode="nmrf", divis_by=16)
            img1, img2 = self.padder.pad(img1, img2)
        else:
            self.padder = None
        H, W = img1.shape[-2:]
        mono_size = (H // 8 * 7, W // 8 * 7)
        mono_input = F.interpolate(img1, size=mono_size, mode="bilinear", align_corners=False, antialias=True)

        inputs["img1"] = img1
        inputs["img2"] = img2
        inputs["image"] = mono_input.sub_(self.mean).div_(self.std)

        return inputs
    
    def postprocessing(self, out, mono_embeds):
        if self.training:
            mono_prediction = F.relu(self.mono_output(mono_embeds))
            out['mono_prediction'] = rearrange(mono_prediction, 'n b h w (hs ws) -> n b (h hs) (w ws)', hs=8)
        else:
            mono_prediction = F.relu(self.mono_output(mono_embeds[-1]))
            out['disp_mono'] = rearrange(mono_prediction, 'b h w (hs ws) -> b (h hs) (w ws)', hs=8)
        
        if self.padder is None:
            return out
        
        if out["disp_pred"].dim() != 4:
            out["disp_pred"] = self.padder.unpad(out["disp_pred"].unsqueeze(1)).squeeze(1)
            out['disp_mono'] = self.padder.unpad(out["disp_mono"].unsqueeze(1)).squeeze(1)
        else:
            out["disp_pred"] = self.padder.unpad(out["disp_pred"])
            out['disp_mono'] = self.padder.unpad(out["disp_mono"])
        return out
    
    def forward(self, inputs):
        inputs = self.prepare_input(inputs)

        depth_prior_raw = self.mono.infer(inputs["image"]).clone()
        depth_prior = self.depth_feature_down(depth_prior_raw).permute(0, 2, 3, 1).contiguous()
        inputs["depth_prior"] = depth_prior

        stereo_outputs = self.stereo_encoder(inputs)
        stereo_embed, stereo_pos_embed = self.stereo_encoder.construct_hypothesis_embedding(
            proposal=stereo_outputs["proposal"].flatten(0, 1).detach(),
            fmap1=stereo_outputs["fmap1_concat_list"][0],
            fmap2=stereo_outputs["fmap2_concat_list"][0],
            fmap1_gw=stereo_outputs["fmap1_gw_list"][0],
            fmap2_gw=stereo_outputs["fmap2_gw_list"][0],
        )

        # align: stage 0
        stereo_embeds, mono_embeds = self.align_0(
            stereo_embed,
            depth_prior,
            stereo_pos_embed
        )
        disp_update = self.infer_head(stereo_embeds)  # [num_aux_layers,B,H,W,N,8*8]
        logit = .25 * self.infer_score_head(stereo_embeds)  # [num_aux_layers,B,H,W,N,8*8]

        # prepare stereo embed for next stage
        proposal = rearrange(stereo_outputs["proposal"].detach(), 'b (h w) n -> b h w n 1', h=disp_update.shape[2])
        disp_0 = F.relu(proposal[None] + disp_update)
        disp_0 = rearrange(disp_0, 'a b h w n (hs ws) -> a b (h hs) (w ws) n', hs=8).contiguous()
        logit = rearrange(logit, 'a b h w n (hs ws) -> a b (h hs) (w ws) n', hs=8)
        _, indices = torch.max(logit[-1], dim=-1, keepdim=True)
        disp_est = torch.gather(disp_0[-1], dim=-1, index=indices).squeeze(-1).detach() * 2  # [B,H,W]
        disp_est = rearrange(disp_est, 'b (h hs) (w ws) -> b h w (hs ws)', hs=4, ws=4)
        # disp_est = torch.median(disp_est.detach(), dim=-1, keepdim=True)[0]
        disp_est = torch.sort(disp_est.detach(), dim=-1)[0][..., 7:8]  # replace median for onnx export
        stereo_embed, stereo_pos_embed = self.stereo_encoder.construct_hypothesis_embedding(
            proposal=disp_est.flatten(0, 2),
            fmap1=stereo_outputs["fmap1_concat_list"][1],
            fmap2=stereo_outputs["fmap2_concat_list"][1],
            fmap1_gw=stereo_outputs["fmap1_gw_list"][1],
            fmap2_gw=stereo_outputs["fmap2_gw_list"][1],
            normalizer=128,
        )

        # align: stage 1
        stereo_embeds, mono_embeds_1 = self.align_1(
            stereo_embed,
            mono_embeds[-1],
            stereo_pos_embed
        )
        disp_update = self.refine_head(stereo_embeds.squeeze(-2))  # [num_aux_layers,B,H,W,4*4]
        disp_1 = F.relu(disp_est[None] + disp_update)
        disp_1 = rearrange(disp_1, 'a b h w (hs ws) -> a b (h hs) (w ws)', hs=4).contiguous() * 4
        disp_final = disp_1[-1]

        results = {
            "coarse_disp": disp_0[-1] * 8,
            "logit": logit[-1],
            "disp_pred": disp_final,
            "proposal": stereo_outputs["proposal"],
            "prob": stereo_outputs["prob"],
        }
        if self.training:
            results['aux_outputs'] = self._set_aux_loss(disp_1, disp_0, logit)

        mono_embeds = torch.cat((mono_embeds, mono_embeds_1), dim=0)

        results = self.postprocessing(results, mono_embeds)

        return results
    
    @torch.jit.unused
    def _set_aux_loss(self, disp_pred, coarse_disp, logit):
        res = []
        for aux, disp_pred_i in enumerate(disp_pred[:-1]):
            res.append(dict(disp_pred=disp_pred_i, aux=aux))
        for aux, (coarse_disp_i, logit_i) in enumerate(zip(coarse_disp[:-1], logit[:-1])):
            res.append(dict(coarse_disp=coarse_disp_i * 8, logit=logit_i, aux=aux))
        return res