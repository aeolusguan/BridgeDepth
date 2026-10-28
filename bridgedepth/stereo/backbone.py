from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .submodule import FeatureFusionBlock, _make_scratch, init_weights


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=False,
    )


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, stride=stride, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)


        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not (stride == 1 and in_planes == planes):
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        identity = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))

        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x + identity)
    

class Backbone(nn.Module):
    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d, normalize_input=True):
        super(Backbone, self).__init__()
        self.normalize_input = normalize_input

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # 1/2
        self.norm1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1, norm_layer=norm_layer) # 1/2
        self.layer2 = self._make_layer(96, stride=2, norm_layer=norm_layer) # 1/4
        self.layer3 = self._make_layer(128, stride=1, norm_layer=norm_layer) # 1/4

        self.conv2 = nn.Conv2d(128, output_dim, 1, 1, 0)

        self.output_dim = output_dim

        self.apply(init_weights)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        layer2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.normalize_input:
            x = 2 * (x / 255.0) - 1.0

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2
        x = self.layer2(x)  # 1/4
        x = self.layer3(x)  # 1/4
        x = self.conv2(x)

        out = [x, F.avg_pool2d(x, kernel_size=2, stride=2)]  # high to low res

        return out
    

class SwinDPT(nn.Module):
    def __init__(
        self, 
        output_dim, 
        use_bn=False,
        stochastic_depth_prob: float = 0.0, 
        _init_weights=True
    ):
        super().__init__()

        from torchvision.models import SwinTransformer

        model = SwinTransformer(
            patch_size=[4, 4],
            embed_dim=96,
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            window_size=[7, 7],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.features = model.features

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norms = nn.ModuleList(
            [norm_layer(96), norm_layer(192), norm_layer(384)]
        )

        self.scratch = _make_scratch(
            [96, 192, 384],
            output_dim,
            groups=1,
            expand=False,
        )

        self.scratch.refinenet1 = _make_fusion_block(output_dim, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(output_dim, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(output_dim, use_bn)

        self.output_dim = output_dim

        self.apply(init_weights)

        if _init_weights:
            from torchvision.models import swin_t, Swin_T_Weights
            pretrained_dict = swin_t(weights=Swin_T_Weights.DEFAULT).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

        self.register_buffer(
            "mean", torch.as_tensor([123.675, 116.28, 103.53], dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.as_tensor([58.395, 57.12, 57.375], dtype=torch.float32)[None, :, None, None]
        )

    def _forward_features(self, x):
        x = (x - self.mean) / self.std

        outs = []
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i in [1, 3, 5]:
                outs.append(x)

        features = []
        for feat, norm in zip(outs, self.norms):
            features.append(norm(feat).permute(0, 3, 1, 2).contiguous())
        return features
    
    def forward(self, x):
        features = self._forward_features(x)

        layer_1, layer_2, layer_3 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)

        path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = [path_1, F.avg_pool2d(path_1, kernel_size=2, stride=2)]  # high to low res

        return out
    

class ConvNextDPT(nn.Module):
    def __init__(
        self, 
        output_dim, 
        use_bn=False,
        stochastic_depth_prob: float = 0.0, 
        _init_weights=True
    ):
        super().__init__()

        from torchvision.models import ConvNeXt
        from torchvision.models.convnext import CNBlockConfig

        model = ConvNeXt(
            block_setting=[
                CNBlockConfig(96, 192, 3),
                CNBlockConfig(192, 384, 3),
                CNBlockConfig(384, None, 9),
            ],
            stochastic_depth_prob=stochastic_depth_prob,
        )
        self.features = model.features

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norms = nn.ModuleList(
            [norm_layer(96), norm_layer(192), norm_layer(384)]
        )

        self.scratch = _make_scratch(
            [96, 192, 384],
            output_dim,
            groups=1,
            expand=False,
        )

        self.scratch.refinenet1 = _make_fusion_block(output_dim, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(output_dim, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(output_dim, use_bn)

        self.output_dim = output_dim

        self.apply(init_weights)

        if _init_weights:
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            pretrained_dict = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).state_dict()
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)

        self.register_buffer(
            "mean", torch.as_tensor([123.675, 116.28, 103.53], dtype=torch.float32)[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.as_tensor([58.395, 57.12, 57.375], dtype=torch.float32)[None, :, None, None]
        )

    def _forward_features(self, x):
        x = (x - self.mean) / self.std

        outs = []
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i in [1, 3, 5]:
                outs.append(x)

        features = []
        for feat, norm in zip(outs, self.norms):
            features.append(norm(feat.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous())
        return features
    
    def forward(self, x):
        features = self._forward_features(x)

        layer_1, layer_2, layer_3 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)

        path_3 = self.scratch.refinenet3(layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = [path_1, F.avg_pool2d(path_1, kernel_size=2, stride=2)]  # high to low res

        return out


def create_backbone(cfg):
    model_type = cfg.BACKBONE.MODEL_TYPE
    if model_type == "resnet":
        if cfg.BACKBONE.NORM_FN == "instance":
            norm_layer = nn.InstanceNorm2d
        elif cfg.BACKBONE.NORM_FN == 'batch':
            norm_layer = nn.BatchNorm2d
        else:
            raise ValueError(f'Invalid backbone normalization type: {cfg.BACKBONE.NORM_FN}')
        backbone = Backbone(cfg.BACKBONE.OUT_CHANNELS, norm_layer)
    elif model_type == "swin":
        backbone = SwinDPT(output_dim=cfg.BACKBONE.OUT_CHANNELS, stochastic_depth_prob=cfg.BACKBONE.DROP_PATH)
    elif model_type == "convnext":
        backbone = ConvNextDPT(output_dim=cfg.BACKBONE.OUT_CHANNELS, stochastic_depth_prob=cfg.BACKBONE.DROP_PATH)
    else:
        raise ValueError(f"Do not find {model_type}")
    
    return backbone
