import torch
from torch import nn

from .stereo import build_criterion as build_stereo_criterion


@torch.no_grad()
def normalize_gt_disparity(disp, mask):
    """Divide each GT disparity map by its valid-pixel mean; invalid pixels are zero.

    ``disp`` and ``mask`` have shape [B,H,W]. Statistics use float32, and an
    image without valid pixels produces an all-zero target.
    """
    valid_disp = torch.where(mask, disp.float(), 0.0)
    count = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
    mean = valid_disp.sum(dim=(-2, -1), keepdim=True) / count
    return valid_disp / mean.clamp_min(1e-6)


def sequence_gt_normalized_loss(predictions, target, mask, loss_gamma=0.9):
    """
    Masked L1 between raw monocular predictions and mean-normalized GT.

    predictions: [N,B,H,W], where N counts refinement stages of one stereo pair
    target:      [B,H,W], GT disparity in pixels
    mask:        [B,H,W], shared by normalization and supervision
    """
    n_predictions = predictions.shape[0]
    
    # Preserve the existing weighting across refinement stages.
    adjust_loss_gamma = loss_gamma**(10/(n_predictions-1)) if n_predictions > 1 else 1.0
    loss = 0.0
    target = normalize_gt_disparity(target, mask)
    valid_count = mask.flatten(1).sum(dim=1).clamp_min(1)
    
    for i in range(n_predictions):
        i_weight = adjust_loss_gamma**(n_predictions - i - 1)
        i_loss = torch.where(mask, (predictions[i] - target).abs(), 0.0)
        i_loss = i_loss.flatten(1).sum(dim=1) / valid_count
        loss += i_weight * i_loss.mean()
    
    assert not torch.isnan(loss)
    return loss
        
        
class BridgeCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.stereo_criterion = build_stereo_criterion(cfg)
        self.loss_gamma = cfg.BRIDGEDEPTH.RECURRENT_LOSS_GAMMA
        self.weight_dict = dict(self.stereo_criterion.weight_dict)
        self.weight_dict.update({'loss_recurrent': cfg.BRIDGEDEPTH.RECURRENT_LOSS_WEIGHT})
        
    def forward(self, outputs, targets, log):
        loss_dict = self.stereo_criterion(outputs, targets, log)
        
        state_predictions = outputs['mono_prediction']
        device = state_predictions.device
        tgt_disp = targets['disp'].to(device)
        valid = targets['valid'].to(device)
        valid = valid & (tgt_disp < self.stereo_criterion.max_disp)
        loss_recurrent = sequence_gt_normalized_loss(state_predictions, tgt_disp, valid, self.loss_gamma)
        
        loss_dict.update({'loss_recurrent': loss_recurrent})
        return loss_dict
        
        
def build_criterion(cfg):
    return BridgeCriterion(cfg)
