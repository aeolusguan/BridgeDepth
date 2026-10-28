import copy
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat


def fourier_coord_embed(coord, N_freqs, normalizer=3.14/512, logscale=True):
    """
    coord: [...]D
    returns:
        [...]dim, where dim=(2*N_freqs+1)*D
    """
    if logscale:
        freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs, device=coord.device)
    else:
        freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
    coord = coord.unsqueeze(-1) * normalizer
    freq_bands = freq_bands[(*((None,) * (len(coord.shape) - 1)), Ellipsis)]
    f_coord = coord * freq_bands
    embed = torch.cat([f_coord.sin(), f_coord.cos(), coord], dim=-1)
    embed = rearrange(embed, '... d n -> ... (d n)')

    return embed


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class Head(nn.Module):
    """ Very simple multi-layer perception (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# Neural Message Passing along self edges of the NMRF graph.
class Attention(nn.Module):
    """
    Hypothesis representation: [B,N,C]
    """
    def __init__(self, dim, qk_dim, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be multiple times of heads {num_heads}'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.q, self.k, self.v = nn.Linear(qk_dim, dim, bias=True), nn.Linear(qk_dim, dim, bias=True), nn.Linear(dim, dim, bias=True)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x, pos_embed):
        """
        x: [B,N,C], embedding of the hypothesis
        pos_embed: [B,N,C'], encoding of the hypothesis disparity
        Returns:
            [B,N,C], aggregated message from self edges
        """
        short_cut = x
        x = self.norm1(x)
        q = k = torch.cat((x, pos_embed), dim=-1)

        # multi-head attention
        q, k, v = self.q(q), self.k(k), self.v(x)
        q, k, v = [rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads) for t in [q, k, v]]
        attn = F.softmax(torch.einsum('bhid, bhjd -> bhij', q, k) * self.scale, dim=-1)
        attn = self.attn_drop(attn)
        x = torch.einsum('bhij, bhjd -> bhid', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        # residual connection
        x = short_cut + self.proj_drop(self.proj(x))

        return x
    

def window_partition(x, window_size):
    """
    x: [B,H,W,N,C]
    Returns:
        (num_windows*B,window_size*window_size*N,C]
    """
    B, H, W, N, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, N, C)
    windows = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous().view(-1, window_size*window_size*N, C)
    return windows


def window_reverse(windows, window_size, H, W, N):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, N, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        N (int): Number of hypothesis
    Returns:
        x: (B, H, W, N, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, N, -1)
    x = x.permute(0, 1, 3, 2, 4, 5, 6).contiguous().view(B, H, W, N, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head positional sensitive self attention (W-MSA).
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        shift_size (int): Shift size for SW-MSA.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override a default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(self, dim, qkv_dim, window_size, shift_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(qkv_dim, 3*dim, bias=qkv_bias)

        # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # define a parameter table of relative position bias
        self.relative_position_enc_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), dim*3))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_enc_table, std=0.02)

    @staticmethod
    def gen_window_attn_mask(window_size, device=torch.device('cuda')):
        """
        Generating attention mask to prevent message passing along self edges.

        Args:
            window_size (tuple[int]): The height, width, and depth (number of candidates) of attention window
        """
        idx = torch.arange(0, window_size[0] * window_size[1], dtype=torch.float32, device=device).view(-1, 1)
        idx = idx.expand(-1, window_size[2]).flatten()
        attn_mask = idx.unsqueeze(-1) - idx.unsqueeze(0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask != 0, 0.0)
        # attn_mask.fill_diagonal_(0.0)
        arange = torch.arange(idx.numel())
        attn_mask[arange, arange] = 0  # replace fill_diagonal_ for onnx export
        return attn_mask
    
    @staticmethod
    def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
        """
        Generating attention mask for shifted window attention, modified from SWin Transformer.

        Args:
            input_resolution (tuple[int]): The height and width of input
            window_size (tuple[int]): The height, width and depth (number of candidates) of window
            shift_size (int): shift size for SW-MSA.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = rearrange(img_mask, 'b (h hs) (w ws) c -> (b h w) (hs ws) c', hs=window_size[0], ws=window_size[1])
        mask_windows = mask_windows.squeeze(-1)  # [num_windows, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float('0.0'))
        attn_mask = repeat(attn_mask, 'b h w -> b (h h2) (w w2)', h2=window_size[2], w2=window_size[2])
        attn_mask = attn_mask + WindowAttention.gen_window_attn_mask(window_size, device).unsqueeze(0)
        return attn_mask
    
    def forward(self, x, attn_mask):
        """
        x:     [num_windows*B, window_size*window_size*N, C']
        mask:  [num_windows, window_size*window_size*N, window_size*window_size*N]
        Returns:
            [num_windows*B, window_size*window_size*N, C]
        """
        B_, L, _ = x.shape
        window_size = self.window_size
        N = L // (window_size[0] * window_size[1])
        C = self.dim

        qkv = (
            self.qkv(x)
            .reshape(B_, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # positional embedding
        rpe = self.relative_position_enc_table[self.relative_position_index.view(-1)].view(
            window_size[0] * window_size[1], window_size[0] * window_size[1], self.num_heads, -1)
        rpe = repeat(rpe, 'i j h c -> (i hs) (j ws) h c', hs=N, ws=N)
        q_rpe, k_rpe, v_rpe = rpe.chunk(3, dim=-1)

        # window attention
        q = q * self.scale
        q_rpe = q_rpe * self.scale
        qk = (q @ k.transpose(-2, -1))  # B head L C @ B head C L --> B head L L
        qr = torch.einsum('bhic,ijhc->bhij', q, k_rpe)
        kr = torch.einsum('bhjc,ijhc->bhij', k, q_rpe)
        attn = qk + qr + kr
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, L, L) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v + torch.einsum('bhij,ijhc->bhic', attn, v_rpe)
        x = x.transpose(1, 2).reshape(B_, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, shift_size={self.shift_size}, num_heads={self.num_heads}'
    

def to_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    
    assert isinstance(x, int)
    return (x, x)


class SwinNMP(nn.Module):
    r"""Swin Message Passing Block.

    Args:
        dim (int): Number of input channels.
        qkv_dim (int): Number of input token channels
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self, dim, qkv_dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, qkv_dim, window_size=to_2tuple(self.window_size), shift_size=shift_size, num_heads=num_heads,
            qk_scale=qk_scale, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def add_pos_embed(self, x, pos_embed):
        x = self.norm1(x)
        # concat latent embedding with position embedding
        x = torch.cat((x, pos_embed), dim=-1)
        return x

    def forward(self, x, pos_embed, attn_mask):
        """
        x: [B,H,W,N,C], hypothesis embedding
        pos_embed: [B,H,W,N,C'], encoding of the underlying disparity
        attn_mask: [num_windows, window_size*window_size*N, window_size*window_size*N],
            attention mask to prevent message passing along self edges.
        Returns: [B,H,W,N,C]
        """
        H, W, N, C = x.shape[1:]
        shortcut = x
        x = self.add_pos_embed(x, pos_embed)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, N, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, N)  # B H' W' N C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, qkv_dim={self.qkv_dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}"
    

class CSWinAttention(nn.Module):
    def __init__(self, dim, idx, split_size=7, num_heads=8, qk_scale=None, attn_drop=0., fused_attn=True):
        """Attention within cross-shaped windows.
        """
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.fused_attn = fused_attn
        self.idx = idx
        self.H_sp = None
        self.W_sp = None

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        x = rearrange(x, 'b (i hs) (j ws) n (h d) -> (b i j) h (hs ws n) d', hs=self.H_sp, ws=self.W_sp, h=self.num_heads)
        return x

    def get_rpe(self, x):
        """
        x: [B,H,W,N,C]
        return:
            rpe: [B,H,W,N,C]
        """
        B, H, W, N, C = x.shape
        rpe_x = x.permute(0, 3, 4, 1, 2).flatten(0, 1)  # BN, C, H, W
        rpe = self.get_v(rpe_x).reshape(B, N, C, H, W).sum(1, keepdim=True)
    
        # prevent the positional embedding of self edges
        self_scaled = rpe_x * self.get_v.weight[None, :, 0, 1, 1, None, None]
        self_scaled = self_scaled.reshape(B, N, C, H, W)
        self_scaled = self_scaled - self_scaled.sum(1, keepdim=True)
    
        rpe = rpe + self_scaled  # B, N, C, H, W
    
        return rpe.permute(0, 3, 4, 1, 2)

    def forward(self, query, key, value):
        """
        query: BHWNC
        key:   BHWNC
        value: BHWNC
        Returns:
            BHWNC
        """
        _, H, W, N, _ = query.shape
        device = query.device

        idx = self.idx
        if idx == -1:
            H_sp, W_sp = H, W
        elif idx == 0:
            H_sp, W_sp = H, self.split_size
        elif idx == 1:
            H_sp, W_sp = self.split_size, W
        else:
            raise RuntimeError(f"ERROR MODE `{idx}` in forward")
        self.H_sp = H_sp
        self.W_sp = W_sp

        # padding for split window
        pad_t = pad_l = 0
        pad_b = (self.H_sp - H % self.H_sp) % self.H_sp
        pad_r = (self.W_sp - W % self.W_sp) % self.W_sp
        Hp = H + pad_b
        Wp = W + pad_r

        if pad_b > 0 or pad_r > 0:
            query = F.pad(query, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            key = F.pad(key, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            value = F.pad(value, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))

        q = self.im2cswin(query)
        k = self.im2cswin(key)
        rpe = self.get_rpe(value)
        v = self.im2cswin(value)

        # Local attention
        window_size = (self.H_sp, self.W_sp, N)
        attn_bias = WindowAttention.gen_window_attn_mask(window_size, device)[None, None, ...]
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = attn + attn_bias
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = rearrange(x, '(b i j) h (hs ws n) d -> b (i hs) (j ws) n (h d)', i=Hp//self.H_sp, j=Wp//self.W_sp, hs=self.H_sp, ws=self.W_sp)
        x = x + rpe
        x = x[:, pad_t:H+pad_t, pad_l:W+pad_l, :, :].contiguous()

        return x


class CSWinNMP(nn.Module):

    def __init__(self, dim, qk_dim, num_heads, split_size=7, mlp_ratio=4., qk_scale=None,
                 attn_drop=0., proj_drop=0., drop_path=0., dropout=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio

        self.q, self.k, self.v = nn.Linear(qk_dim, dim, bias=True), nn.Linear(qk_dim, dim, bias=True), nn.Linear(dim, dim, bias=True)
        self.norm1 = norm_layer(dim)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attns = nn.ModuleList([
            CSWinAttention(
                dim//2, idx=i, split_size=split_size, num_heads=num_heads//2, qk_scale=qk_scale,
                attn_drop=attn_drop)
            for i in range(2)])
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=dropout)
        self.norm2 = norm_layer(dim)

    def add_context(self, x, context):
        x = self.norm1(x)
        q = k = torch.cat([x, context], dim=-1)
        return q, k, x

    def forward(self, x, context):
        """
        x: [B,H,W,N,C], embedding of the disparity
        context: [B,H,W,N,C'], visual context of the disparity
        """
        shortcut = x

        q, k, v = self.add_context(x, context)
        query, key, value = self.q(q), self.k(k), self.v(v)

        # cross shaped window attention
        x1 = self.attns[0](query[..., :self.dim // 2], key[..., :self.dim // 2], value[..., :self.dim // 2])
        x2 = self.attns[1](query[..., self.dim // 2:], key[..., self.dim // 2:], value[..., self.dim // 2:])
        x = torch.cat([x1, x2], dim=-1)
        x = self.proj_drop(self.proj(x))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class Propagation(nn.Module):
    """Label seed propagation"""
    def __init__(self, embed_dim, cost_group, prop_layer, num_layers, norm=None):
        super().__init__()
        self.corr_encoder = nn.Sequential(
            nn.Linear(cost_group*9, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
        )
        self.proj = nn.Linear(embed_dim+31, embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.layers = _get_clones(prop_layer, num_layers)
        self.norm = norm if norm is not None else nn.Identity()

    @staticmethod
    def sample_corr(corr_volume, label_seed):
        """
        corr_volume: [BHW,G,D], groupwise cost volume
        label_seed: [BHW,num_seed], integer disparity modals
        return:
            [BHW,num_seed,G*9], sampled cost values of the label seed
        """
        G, D = corr_volume.shape[-2:]
        num_seed = label_seed.shape[1]
        offset = torch.arange(-4, 5, device=corr_volume.device, dtype=label_seed.dtype)
        idx = label_seed[..., None] + offset.view(1, 1, -1)  # [BHW,num_seed,9]
        idx = torch.clamp(idx, min=0, max=D-1)
        idx = idx.reshape(-1, 1, 9*num_seed).repeat(1, G, 1)
        corr = torch.gather(corr_volume, dim=-1, index=idx)
        corr = rearrange(corr, 'b h (n c) -> b n (h c)', n=num_seed)
        return corr

    def forward(self, corr_volume, label_seed, context, geo_prior):
        """
        corr_volume: [BHW,G,D], groupwise cost volume
        label_seed:  [BHW,num_seed], integer disparity modals
        context:     [B,H,W,C], visual context
        geo_prior:   [B,H,W,C], depth memory prior
        Returns:
            refined embed: [B,H,W,num_seed,C], label seed: [BHW,num_seed]
        """
        # extract cost of label seeds
        N = label_seed.shape[-1]
        H, W = context.shape[1:3]
        corr = Propagation.sample_corr(corr_volume, label_seed)
        corr = self.corr_encoder(corr)
        corr = rearrange(corr, '(b h w) n c -> b h w n c', h=H, w=W)

        # position encoding
        label_seed = label_seed.float()
        pos_embed = fourier_coord_embed(label_seed.reshape(-1, H, W, N, 1), N_freqs=15, normalizer=3.14 / 64)

        # hypothesis embedding
        geo_prior = geo_prior.unsqueeze(-2).repeat(1, 1, 1, N, 1)
        x = self.proj(torch.cat((corr, pos_embed, geo_prior), dim=-1))
        
        context = repeat(context, 'b h w c -> b h w n c', n=N)
        # attentional propagation
        for layer in self.layers:
            x = layer(x, context)

        return self.norm(x), label_seed


class Inference(nn.Module):
    """Neural MRF Inference"""
    def __init__(self, cost_group, dim, layers, norm, return_intermediate=False):
        super().__init__()

        self.ffn = Mlp(dim+cost_group, dim, dim)
        self.dim = dim
        self.layers = layers
        self.norm = norm if norm is not None else nn.Identity()
        self.cost_group = cost_group
        self.return_intermediate = return_intermediate

    @staticmethod
    def sample_fmap(fmap, disp_sample, radius=4):
        """
        fmap: [B,C,H,W]
        disp_sample: tensor of dim [BHW,num_disp], disparity samples
        radius(int): 2*radius+1 samples will be sampled for each sample
        return:
            sampled fmap feature of dim [B,C,H,W,num_disp*(2*radius+1)]
        """
        bs, _, ht, wd = fmap.shape
        num_disp = disp_sample.shape[1]
        device = fmap.device
        with torch.no_grad():
            offset = torch.arange(-radius, radius+1, dtype=disp_sample.dtype, device=disp_sample.device).view(1, 1, -1)
            grid_x = disp_sample[..., None] + offset  # [B*H*W, num_disp, 2*r+1]
            grid_x = grid_x.reshape(bs, ht, wd, -1)  # [B, H, W, num_disp*(2*r+1)]
            grid_y = torch.zeros_like(grid_x)
            xs = torch.arange(0, wd, device=device, dtype=torch.float32).view(1, wd).expand(ht, wd)
            ys = torch.arange(0, ht, device=device, dtype=torch.float32).view(ht, 1).expand(ht, wd)
            grid = torch.stack((xs, ys), dim=-1).reshape(1, ht, wd, 1, 2)
            grid = grid + torch.stack((-grid_x, grid_y), dim=-1)  # [B, H, W, num_disp*(2*r+1), 2]
            grid[..., 0] = 2 * grid[..., 0].clone() / (wd - 1) - 1
            grid[..., 1] = 2 * grid[..., 1].clone() / (ht - 1) - 1
            grid = grid.reshape(bs, ht, -1, 2)  # [B, H, W*num_disp*(2*r+1), 2]
        samples = F.grid_sample(fmap, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return samples.reshape(bs, -1, ht, wd, num_disp*(2*radius+1))

    def corr(self, fmap1, warped_fmap2, num_disp):
        """
        fmap1: [B,C,H,W]
        warped_fmap2: [B,C,H,W,num_disp]
        Returns:
            local cost: [B,H,W,num_disp,G]
        """
        fmap1 = rearrange(fmap1, 'b (g d) h w -> b g d h w', g=self.cost_group)
        warped_fmap2 = rearrange(warped_fmap2, 'b (g d) h w n -> b g d h w n', g=self.cost_group)
        corr = (fmap1.unsqueeze(-1) * warped_fmap2).mean(dim=2)  # [B,G,H,W,num_disp]
        corr = rearrange(corr, 'b g h w n -> b h w n g', n=num_disp)
        return corr
    
    def construct_label_embedding(self, labels, fmap1, fmap2, fmap1_gw, fmap2_gw, normalizer=64):
        H, W, N = fmap1.shape[2], fmap1.shape[3], labels.shape[-1]
        warped_fmap2_gw = self.sample_fmap(fmap2_gw, labels, radius=0)  # [B,C,H,W,N]
        corr = self.corr(fmap1_gw, warped_fmap2_gw, N)  # [B,H,W,N,G]
        warped_fmap2 = self.sample_fmap(fmap2, labels, radius=0)  # [B,C,H,W,N]
        fmap1 = repeat(fmap1, 'b c h w -> b c h w n', n=N)
        feat_concat = torch.cat((fmap1, warped_fmap2), dim=1)
        feat_concat = rearrange(feat_concat, 'b c h w n -> b h w n c')
        x = self.ffn(torch.cat((feat_concat, corr), dim=-1))
        labels = rearrange(labels, '(b h w) n -> b h w n', h=H, w=W)
        pos_embed = fourier_coord_embed(labels.unsqueeze(-1), N_freqs=15, normalizer=3.14 / normalizer)
        return x, pos_embed
    
    def forward(self, labels, fmap1, fmap2, fmap1_gw, fmap2_gw, memory):
        """
        labels: [BHW,num_disp], candidate labels (disparity)
        fmap1: [B,C,H,W]
        fmap2: [B,C,H,W]
        fmap1_gw: [B,C,H,W]
        fmap2_gw: [B,C,H,W]
        memory: Shared Memory Interface
        Returns:
            [B,H,W]
        """
        ht, wd = fmap1.shape[2:]
        N = labels.shape[-1]
        device = labels.device
        label_embedding, abs_encoding = self.construct_label_embedding(labels, fmap1, fmap2, fmap1_gw, fmap2_gw)

        # pad input to multiple times of window_size (assume all swin blocks have the same window size)
        window_size = self.layers[0].window_size
        H_pad = (window_size - ht % window_size) % window_size
        W_pad = (window_size - wd % window_size) % window_size
        pad_t = H_pad // 2
        pad_b = H_pad - pad_t
        pad_l = W_pad // 2
        pad_r = W_pad - pad_l
        hp = ht + H_pad
        wp = wd + W_pad

        if pad_r > 0 or pad_b > 0:
            label_embedding = F.pad(label_embedding, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            abs_encoding = F.pad(abs_encoding, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            memory.pad(pad_t, pad_b, pad_l, pad_r)

        # hack implementation to cache attention mask
        window_size = (window_size, window_size, N)
        window_attn_mask = WindowAttention.gen_window_attn_mask(window_size, device=device)[None]
        attn_mask = [window_attn_mask]
        if len(self.layers) >= 2:
            shift_size = self.layers[1].shift_size
            input_resolution = (hp, wp)
            shifted_window_attn_mask = WindowAttention.gen_shift_window_attn_mask(
                input_resolution, window_size, shift_size, device
            )
            attn_mask.append(shifted_window_attn_mask)

        intermediate = []
        states = []
        return_intermediate = self.return_intermediate and self.training
        for idx, layer in enumerate(self.layers):
            label_embedding = memory.readout(label_embedding, abs_encoding)
            label_embedding = layer(label_embedding, abs_encoding, attn_mask[idx%2])
            memory.update(label_embedding, abs_encoding)
            if return_intermediate:
                intermediate.append(self.norm(label_embedding[:, pad_t:ht+pad_t, pad_l:wd+pad_l, :, :]))
                states.append(memory.memory_unpad(pad_t, pad_l, ht, wd))

        label_embedding = label_embedding[:, pad_t:ht+pad_t, pad_l:wd+pad_l, :, :]
        memory.unpad(pad_t, pad_l, ht, wd)
        if self.norm is not None:
            label_embedding = self.norm(label_embedding)
            if return_intermediate:
                intermediate.pop()
                intermediate.append(label_embedding)

        if return_intermediate:
            return torch.stack(intermediate), states

        return label_embedding.unsqueeze(0), [memory.memory]


class Refinement(Inference):
    @staticmethod
    def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
        """
        input_resolution (tuple[int]): The height and width of input
        window_size (tuple[int]): The height, width and depth of window
        shift_size (int): Shift size for SW-MSA.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = rearrange(img_mask, 'b (h hs) (w ws) c -> (b h w) (hs ws) c', hs=window_size[0], ws=window_size[1])
        mask_windows = mask_windows.squeeze(-1)  # [num_windows, window_size*window_size]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf')).masked_fill(attn_mask == 0, float('0.0'))
        return attn_mask

    def forward(self, labels, fmap1, fmap2, fmap1_gw, fmap2_gw, memory):
        """
        labels: [B,H,W]
        fmap1: [B,C,H,W]
        fmap2: [B,C,H,W]
        fmap1_gw: [B,C,H,W]
        fmap2_gw: [B,C,H,W]
        """
        ht, wd = fmap1.shape[2:]
        device = labels.device
        labels = labels.reshape(-1, 1)
        label_embedding, abs_encoding = self.construct_label_embedding(labels, fmap1, fmap2, fmap1_gw, fmap2_gw, normalizer=128)

        # pad input to multiple times of window_size (assume all swin blocks have the same window size)
        window_size = self.layers[0].window_size
        H_pad = (window_size - ht % window_size) % window_size
        W_pad = (window_size - wd % window_size) % window_size
        pad_t = H_pad // 2
        pad_b = H_pad - pad_t
        pad_l = W_pad // 2
        pad_r = W_pad - pad_l
        hp = ht + H_pad
        wp = wd + W_pad

        if pad_r > 0 or pad_b > 0:
            label_embedding = F.pad(label_embedding, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            abs_encoding = F.pad(abs_encoding, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            memory.pad(pad_t, pad_b, pad_l, pad_r)

        # hack implementation to cache attention mask
        window_size = (window_size, window_size)
        attn_mask = [None]
        if len(self.layers) >= 2:
            shift_size = self.layers[1].shift_size
            input_resolution = (hp, wp)
            shifted_window_attn_mask = self.gen_shift_window_attn_mask(
                input_resolution, window_size, shift_size, device
            )
            attn_mask.append(shifted_window_attn_mask)

        intermediate = []
        states = []
        return_intermediate = self.return_intermediate and self.training
        for idx, layer in enumerate(self.layers):
            label_embedding = memory.readout(label_embedding, abs_encoding)
            label_embedding = layer(label_embedding, abs_encoding, attn_mask[idx % 2])
            memory.update(label_embedding, abs_encoding)
            if return_intermediate:
                intermediate.append(self.norm(label_embedding[:, pad_t:ht+pad_t, pad_l:wd+pad_l, :, :]))
                states.append(memory.memory_unpad(pad_t, pad_l, ht, wd))

        label_embedding = label_embedding[:, pad_t:ht+pad_t, pad_l:wd+pad_l, :, :]
        if self.norm is not None:
            label_embedding = self.norm(label_embedding)
            if return_intermediate:
                intermediate.pop()
                intermediate.append(label_embedding)

        if return_intermediate:
            return torch.stack(intermediate).squeeze(-2), states

        return label_embedding.unsqueeze(0).squeeze(-2), [memory.memory_unpad(pad_t, pad_l, ht, wd)]


class PropagationLayer(nn.Module):
    def __init__(self, embed_dim, mlp_ratio, context_dim, split_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super().__init__()

        # self attention
        act_layer = _get_activation_fn(activation)
        # concat seed embedding with visual context when linearly projecting to
        # query and key since visually similar pixel tends to have coherent disparities
        qk_dim = embed_dim + context_dim
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.nmp = CSWinNMP(embed_dim, qk_dim, num_heads=n_heads, split_size=split_size, mlp_ratio=mlp_ratio,
                            attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path, dropout=dropout,
                            act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, context):
        """
        x:          B,H,W,N,C
        context:    B,H,W,N,C
        Returns:
            BHW,N,C
        """
        # self attention
        x = self.nmp(x, context=context)
        return x


class InferenceLayer(nn.Module):
    def __init__(self, dim, mlp_ratio, window_size, shift_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super().__init__()

        # self attention
        act_layer = _get_activation_fn(activation)
        qk_dim = dim + 31
        self.window_size = window_size
        self.shift_size = shift_size
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # attend to proposals of the same pixel to suppress non-accurate proposals
        self.self_nmp = Attention(dim, qk_dim, n_heads, attn_drop=attn_drop, proj_drop=proj_drop,
                                  drop_path=drop_path, dropout=dropout)
        # attend to neighbor pixels to extract feature
        self.nmp = SwinNMP(dim, qk_dim, num_heads=n_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                           drop_path=drop_path, drop=dropout, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, pos_embed, attn_mask):
        """
        x: B,H,W,N,C
        pos_embed: B,H,W,N,C
        """
        H, W = x.shape[1:3]
        x = self.self_nmp(x.flatten(0, 2), pos_embed.flatten(0, 2))
        x = rearrange(x, '(b h w) n c -> b h w n c', h=H, w=W)
        x = self.nmp(x, pos_embed=pos_embed, attn_mask=attn_mask)
        return x


class RefinementLayer(nn.Module):
    def __init__(self, dim, mlp_ratio, window_size, shift_size, n_heads,
                 activation="gelu", attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
        super().__init__()

        act_layer = _get_activation_fn(activation)
        qk_dim = dim + 31
        self.window_size = window_size
        self.shift_size = shift_size
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.nmp = SwinNMP(dim, qk_dim, num_heads=n_heads, window_size=window_size,
                           shift_size=shift_size, mlp_ratio=mlp_ratio, attn_drop=attn_drop,
                           drop_path=drop_path, drop=dropout, act_layer=act_layer, norm_layer=norm_layer)

    def forward(self, x, pos_embed, attn_mask):
        """
        x:        [B,H,W,1,C]
        """
        x = self.nmp(x, pos_embed=pos_embed, attn_mask=attn_mask)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
