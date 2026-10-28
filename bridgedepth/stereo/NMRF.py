import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange, repeat

from bridgedepth.config import configurable
from .backbone import create_backbone
from .DPN import DPN
from .submodule import build_correlation_volume
from .NMP import fourier_coord_embed, Mlp


class NMRF(nn.Module):
    @configurable
    def __init__(self,
                 backbone,
                 dpn,
                 num_proposals,
                 max_disp,
                 infer_embed_dim):
        """
        aux_loss: True if auxiliary intermediate losses (losses at each encoder/decoder layer)
        """
        super().__init__()
        self.num_proposals = num_proposals
        self.max_disp = max_disp

        feat_dim = backbone.output_dim
        self.concatconv = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        )
        self.gw = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0, bias=False))

        self.ffn = Mlp(2 * 64 + 32, infer_embed_dim, infer_embed_dim)
        
        # init weights
        self.apply(self._init_weights)

        self.dpn = dpn
        self.backbone = backbone

        # to keep track of which device the nn.Module is on
        self.register_buffer("device_indicator_tensor", torch.empty(0))

    @classmethod
    def from_config(cls, cfg):
        # backbone
        backbone = create_backbone(cfg)

        # disparity proposal network
        dpn = DPN(cfg)

        return {
            "backbone": backbone,
            "dpn": dpn,
            "num_proposals": cfg.DPN.NUM_PROPOSALS,
            "max_disp": cfg.DPN.MAX_DISP,
            "infer_embed_dim": cfg.NMP.INFER_EMBED_DIM,
        }
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_tensor.device
    
    @staticmethod
    def sample_fmap(fmap, disp):
        """
        fmap: [B,C,H,W]
        disp: tensor of dim [BHW,num_disp], disparity proposals
        return:
            sampled fmap feature of dim [B,C,H,W,num_disp]
        """
        bs, _, ht, wd = fmap.shape
        num_disp = disp.shape[1]
        device = fmap.device
        with torch.no_grad():
            grid_x = disp.reshape(bs, ht, wd, -1)  # [B,H,W,num_disp]
            grid_y = torch.zeros_like(grid_x)
            xs = torch.arange(0, wd, device=device, dtype=torch.float32).view(1, wd).expand(ht, wd)
            ys = torch.arange(0, ht, device=device, dtype=torch.float32).view(ht, 1).expand(ht, wd)
            grid = torch.stack((xs, ys), dim=-1).reshape(1, ht, wd, 1, 2)
            grid = grid + torch.stack((-grid_x, grid_y), dim=-1)  # [B,H,W,num_disp,2]
            grid[..., 0] = 2 * grid[..., 0].clone() / (wd - 1) - 1
            grid[..., 1] = 2 * grid[..., 1].clone() / (ht - 1) - 1
            grid = grid.reshape(bs, ht, -1, 2)  # [B,H,W*num_disp,2]
        feats = F.grid_sample(fmap, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return feats.reshape(bs, -1, ht, wd, num_disp)
    
    def corr(self, fmap1, warped_fmap2, num_disp):
        """
        fmap1: [B,C,H,W]
        warped_fmap2: [B,C,H,W,num_disp]
        Returns:
            local cost: [B,H,W,num_disp,G]
        """
        fmap1 = rearrange(fmap1, 'b (g d) h w -> b g d h w', g=32)
        warped_fmap2 = rearrange(warped_fmap2, 'b (g d) h w n -> b g d h w n', g=32)
        corr = (fmap1.unsqueeze(-1) * warped_fmap2).mean(dim=2)  # [B,G,H,W,num_disp]
        corr = rearrange(corr, 'b g h w n -> b h w n g', n=num_disp)
        return corr
    
    def construct_hypothesis_embedding(self, proposal, fmap1, fmap2, fmap1_gw, fmap2_gw, normalizer=64):
        H, W, N = fmap1.shape[2], fmap1.shape[3], proposal.shape[-1]
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, proposal)  # [B,C,H,W,N]
        corr = self.corr(fmap1_gw, warped_fmap2_gw, N)  # [B,H,W,N,G]
        warped_fmap2 = self.sample_fmap(fmap2, proposal)  # [B,C,H,W,N]
        fmap1 = repeat(fmap1, 'b c h w -> b c h w n', n=N)
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = rearrange(feat_concat, 'b c h w n -> b h w n c')
        x = self.ffn(torch.cat((feat_concat, corr), dim=-1))
        proposal = rearrange(proposal, '(b h w) n -> b h w n', h=H, w=W)
        pos_embed = fourier_coord_embed(proposal.unsqueeze(-1), N_freqs=15, normalizer=3.14 / normalizer)
        return x, pos_embed

    def extract_feature(self, img1, img2):
        img_batch = torch.cat((img1, img2), dim=0)  # [2B,C,H,W]
        features = self.backbone(img_batch)

        # reverse resolution from low to high
        features = features[::-1]

        # split to list of tuple, res from low to high
        features = [torch.chunk(feature, 2, dim=0) for feature in features]

        feature1, feature2 = map(list, zip(*features))

        return feature1, feature2
    
    def forward(self, input):
        """
        It returns a dict with the following elements:
            - "proposals": disparity proposals, tensor of dim [n, B*H/2*W/2, num_proposals]
            - "prob": disparity candidate probability, tensor of dim [B*H/2*W/2, W/2]
            - "disp": disparity prediction, tensor of dim [B, H, W]
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                             dictionaries containing the four above keys for each intermediate layer.
        """
        img1 = input["img1"]
        img2 = input["img2"]
        
        # We assume the input padding is not needed during training by setting adequate crop size
        fmap1_list, fmap2_list = self.extract_feature(img1, img2)

        # proposal extraction
        depth_prior = input['depth_prior']
        cost_volume = build_correlation_volume(fmap1_list[0], fmap2_list[0], self.max_disp // 8, self.dpn.cost_group)  # [B,G,D,H,W]
        prob, labels = self.dpn(cost_volume, fmap1_list[0], depth_prior)

        # construct hypothesis embedding
        fmap1_concat_list = [self.concatconv(x) for x in fmap1_list]
        fmap2_concat_list = [self.concatconv(x) for x in fmap2_list]
        fmap1_gw_list = [self.gw(x) for x in fmap1_list]
        fmap2_gw_list = [self.gw(x) for x in fmap2_list]

        proposal = labels.reshape(img1.shape[0], -1, self.num_proposals)
        out = {
            "prob": prob,
            "proposal": proposal,
            "fmap1_concat_list": fmap1_concat_list,
            "fmap2_concat_list": fmap2_concat_list,
            "fmap1_gw_list": fmap1_gw_list,
            "fmap2_gw_list": fmap2_gw_list,
        }
        
        return out
    

class Criterion(nn.Module):
    """ This class computes the loss for disparity proposal extraction.
    The process happens in two steps:
        1) we compute a one-to-one matching between ground truth disparities and the outputs of the model
        2) we supervise each output to be closer to the ground truth disparity it was matched to

    Note: to avoid trivial solution, we add a prior term in the loss computation that we favor positive output.
    """
    def __init__(self, weight_dict, cfg):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
        """
        super().__init__()
        if cfg.SOLVER.AUX_LOSS:
            # TODO
            pass
        self.weight_dict = weight_dict
        self.max_disp = cfg.SOLVER.MAX_DISP
        assert cfg.SOLVER.LOSS_TYPE in ['L1', 'SMOOTH_L1'], f"unrecognized loss type {cfg.SOLVER.LOSS_TYPE}"
        if cfg.SOLVER.LOSS_TYPE == "SMOOTH_L1":
            self.loss_fn = F.smooth_l1_loss
        else:
            self.loss_fn = F.l1_loss

    def loss_prop(self, disp_prop, gt_disp):
        """
        disp_prop: [B,hw,N]
        gt_disp: [B,H,W], where H=8*h, W=8*w
        """
        tgt_disp = gt_disp.clone()
        # ground truth modes larger than 320 are ignored in following matching
        tgt_disp[tgt_disp >= 320] = 0
        tgt_disp = rearrange(tgt_disp, 'b (h m) (w n) -> b (h w) (m n)', m=8, n=8)
        dist = (tgt_disp[:, :, :, None] - disp_prop[:, :, None, :]).abs()
        _, indices = torch.min(dist, dim=-1, keepdim=False)
        src_disp = torch.gather(disp_prop, dim=-1, index=indices)
    
        mask = (tgt_disp > 0) & (tgt_disp < self.max_disp)
        total_gts = torch.sum(mask)
        # disparity loss for matched predictions
        loss_disp = F.smooth_l1_loss(src_disp[mask], tgt_disp[mask], reduction='sum')
        losses = {'loss_prop': loss_disp / (total_gts + 1e-6)}
    
        return losses
    
    @staticmethod
    def loss_init(prob, gt_disp):
        nd = prob.shape[-1]
        bs, ht, wd = gt_disp.shape
        gt_disp = torch.clamp(gt_disp, min=0)
        valid = (gt_disp > 0) & (gt_disp < 320)

        ref = torch.arange(0, wd, dtype=torch.int64, device=prob.device).reshape(1, 1, -1).repeat(bs, ht, 1)
        coord = ref - gt_disp  # corresponding coordinate in the right view
        valid = torch.logical_and(valid, coord >= 0)  # correspondence should within image boundary

        # scale ground-truth disparities
        tgt_disp = gt_disp / 8

        weights = torch.ones_like(tgt_disp)
        weights[~valid] = 0

        tgt_disp = rearrange(tgt_disp, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)
        weights = rearrange(weights, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)
        valid = rearrange(valid, 'b (h m) (w n) -> (b h w) (m n)', m=8, n=8)

        lower_bound = torch.floor(tgt_disp).to(torch.int64)
        high_bound = lower_bound + 1
        high_prob = tgt_disp - lower_bound
        lower_bound = torch.clamp(lower_bound, max=nd - 1)
        high_bound = torch.clamp(high_bound, max=nd - 1)

        lower_prob = (1 - high_prob) * weights
        high_prob = high_prob * weights

        label = torch.zeros_like(prob)
        label.scatter_reduce_(dim=-1, index=lower_bound, src=lower_prob, reduce="sum")
        label.scatter_reduce_(dim=-1, index=high_bound, src=high_prob, reduce="sum")

        # normalize weights
        normalizer = torch.clamp(torch.sum(label, dim=-1, keepdim=True), min=1e-3)
        label = label / normalizer

        mask = label > 0
        log_prob = -(torch.log(torch.clamp(prob[mask], min=1e-6)) * label[mask]).sum()
        valid_pixs = (valid.float().sum(dim=-1) > 0).sum()

        losses = {'init': log_prob / (valid_pixs + 1e-6)}
        assert not torch.any(torch.isnan(losses['init']))
        return losses
    
    def loss_coarse(self, disp_pred, logits_pred, disp_gt):
        mask = (disp_gt > 0) & (disp_gt < self.max_disp)
        prob = F.softmax(logits_pred, dim=-1)
        disp_gt = disp_gt.unsqueeze(-1).expand_as(disp_pred)
        error = self.loss_fn(disp_pred, disp_gt, reduction='none')
        if torch.any(mask):
            loss = torch.sum(prob * error, dim=-1, keepdim=False)[mask].mean()
        else:  # dummy loss
            loss = F.smooth_l1_loss(disp_pred, disp_pred.detach(), reduction='mean') + F.smooth_l1_loss(logits_pred, logits_pred.detach(), reduction='mean')

        return {"loss_coarse_disp": loss}
    
    def loss_disp(self, disp_pred, disp_gt):
        mask = (disp_gt > 0) & (disp_gt < self.max_disp)
        if torch.any(mask):
            loss = self.loss_fn(disp_pred[mask], disp_gt[mask], reduction='mean')
        else:
            loss = F.smooth_l1_loss(disp_pred, disp_pred.detach(), reduction='mean')

        return {"loss_disp": loss}
    
    def forward(self, outputs, targets, log=True):
        """This performs the loss computation.
        outputs: dict of tensors, see the output specification of the model for the format
        targets: dict of tensors, the expected keys in each dict depends on the losses applied.
            - "disp": [batch_size,H_I,W_I],
            - "valid": boolean tensor [batch_size, H_I, W_I] valid mask of `disp`.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        prob = outputs_without_aux['prob']  # [BHW,D]
        disp_prop = outputs_without_aux['proposal'] * 8  # [B,HW,N]
        disp = outputs_without_aux['disp_pred']
        device = disp.device

        tgt_disp = targets['disp'].to(device)
        valid = targets['valid'].to(device)
        tgt_disp[~valid] = 0

        losses = self.loss_prop(disp_prop, tgt_disp)
        losses.update(self.loss_init(prob, tgt_disp))
        losses.update(self.loss_coarse(outputs_without_aux['coarse_disp'], outputs_without_aux['logit'], tgt_disp))
        losses.update(self.loss_disp(disp, tgt_disp))

        if log:
            valid = (tgt_disp > 0) & (tgt_disp < self.max_disp)
            err = torch.abs(disp - tgt_disp)
            losses['epe_train'] = err[valid].mean()

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for aux_outputs in outputs['aux_outputs']:
                if 'logit' in aux_outputs:
                    l_dict = self.loss_coarse(aux_outputs['coarse_disp'], aux_outputs['logit'], tgt_disp)
                else:
                    l_dict = self.loss_disp(aux_outputs['disp_pred'], tgt_disp)

                aux = aux_outputs['aux']
                l_dict = {k + f'_{aux}_aux': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
    

def build_criterion(cfg):
    weight_dict = {'loss_prop': 1.0, 'init': 1.0}
    assert len(cfg.SOLVER.LOSS_WEIGHTS) == cfg.NMP.NUM_INFER_LAYERS + cfg.NMP.NUM_REFINE_LAYERS
    if cfg.SOLVER.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.NMP.NUM_INFER_LAYERS + cfg.NMP.NUM_REFINE_LAYERS-1):
            if i < cfg.NMP.NUM_INFER_LAYERS - 1:
                aux_weight_dict.update({f'loss_coarse_disp_{i}_aux': cfg.SOLVER.LOSS_WEIGHTS[i]})
            elif i == cfg.NMP.NUM_INFER_LAYERS - 1:
                weight_dict.update({'loss_coarse_disp': cfg.SOLVER.LOSS_WEIGHTS[i]})
            else:
                aux_weight_dict.update({f'loss_disp_{i-cfg.NMP.NUM_INFER_LAYERS}_aux': cfg.SOLVER.LOSS_WEIGHTS[i]})
        weight_dict.update(aux_weight_dict)
    weight_dict.update({'loss_disp': cfg.SOLVER.LOSS_WEIGHTS[-1]})
    criterion = Criterion(weight_dict, cfg)

    return criterion
