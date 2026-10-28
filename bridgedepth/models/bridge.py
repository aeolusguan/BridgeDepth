"""Token adapters between FILT3R patches and BridgeDepth alignment features."""

from torch import Tensor, nn
import torch.nn.functional as F


class ImageAdapter(nn.Module):
    """Exchange residuals between the native patch grid and the 1/8 bridge grid."""

    def __init__(self, image_dim: int, bridge_dim: int):
        super().__init__()
        self.to_mono = nn.Linear(image_dim, bridge_dim)
        self.to_filt = nn.Linear(bridge_dim, image_dim)
        nn.init.zeros_(self.to_filt.weight)
        nn.init.zeros_(self.to_filt.bias)

    def read(self, tokens: Tensor, patch_shape: tuple[int, int]) -> Tensor:
        height, width = patch_shape
        feature = self.to_mono(tokens).transpose(1, 2).reshape(
            tokens.shape[0], -1, height, width)
        feature = F.interpolate(feature, size=(height * 2, width * 2),
                                mode="bilinear", align_corners=False)
        return feature.permute(0, 2, 3, 1).contiguous()

    def write(self, residual: Tensor, patch_shape: tuple[int, int]) -> Tensor:
        residual = self.to_filt(residual).permute(0, 3, 1, 2)
        residual = F.interpolate(residual, size=patch_shape,
                                 mode="bilinear", align_corners=False)
        return residual.flatten(2).transpose(1, 2)
