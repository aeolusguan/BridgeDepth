import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from einops import rearrange, repeat

from .stereo.NMP import WindowAttention, InferenceLayer, RefinementLayer, to_2tuple, Mlp, DropPath


class CrossAttention(nn.Module):
    """ Window based multi-head positional sensitive cross attention (W-MSA).

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the query window.
        stride (int): The downsample stride of query with respect to key/value.
        num_heads (int): Number of attention heads.
        qk_scale (float | None, optional): Override a default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    """

    def __init__(
        self, 
        dim,
        q_dim,
        kv_dim,
        window_size, 
        stride, 
        num_heads, 
        qkv_bias=True,
        proj_bias=True,
        qk_scale=None, 
        attn_drop=0.,
        proj_drop=0.,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.stride = stride
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        assert (window_size[0] * stride) % 1 == 0.0
        assert (window_size[1] * stride) % 1 == 0.0
        
        # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # define a parameter table of relative position bias
        if stride > 1:
            invs = 1
            self.relative_position_enc_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * stride * stride, dim*3))  # stride*(2*Wh-1) * stride(2*Ww-1), nH
            ww = (2 * window_size[1] - 1) * stride
        else:
            assert 1 % stride == 0
            invs = int(1 / stride)
            self.relative_position_enc_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - invs) * (2 * window_size[1] - invs), dim*3))
            ww = 2 * window_size[1] - invs

        # get pair-wise relative position index for each token inside the window
        coords_h_query = (torch.arange(self.window_size[0]) + 0.5) * stride
        coords_h_key = torch.arange(self.window_size[0] * stride) + 0.5
        coords_w_query = (torch.arange(self.window_size[1]) + 0.5) * stride
        coords_w_key = torch.arange(self.window_size[1] * stride) + 0.5
        coords_query = torch.stack(torch.meshgrid([coords_h_query, coords_w_query]))  # 2, Wh, Ww
        coords_query_flatten = torch.flatten(coords_query, 1)  # 2, Wh*Ww
        coords_key = torch.stack(torch.meshgrid([coords_h_key, coords_w_key]))  # 2, Wh*stride, Ww*stride
        coords_key_flatten = torch.flatten(coords_key, 1)  # 2, Wh*Ww*stride*stride
        relative_coords = coords_query_flatten[:, :, None] - coords_key_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww*stride*stride
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww*stride*stride, 2
        relative_coords[:, :, 0] += stride * (self.window_size[0] - 0.5) - 0.5  # shift to start from 0
        relative_coords[:, :, 1] += stride * (self.window_size[1] - 0.5) - 0.5
        relative_coords *= invs
        relative_coords[:, :, 0] *= ww
        relative_position_index = relative_coords.sum(-1).long()  # Wh*Ww, Wh*Ww*stride*stride
        self.register_buffer("relative_position_index", relative_position_index)

        self.projq = nn.Linear(q_dim, dim, bias=qkv_bias)
        self.projkv = nn.Linear(kv_dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.init_weights()
        
    def init_weights(self):
        trunc_normal_(self.relative_position_enc_table, std=0.02)
        
        for module in self.children():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def window_partition(self, x, query=False):
        """
        x: [B,H,W,N,C]
        Returns:
            [B*num_windows,num_heads,window_size*window_size*N,head_dim]
        """
        stride = 1 if query else self.stride
        x = rearrange(x, 'b (i hs) (j ws) n (h d) -> (b i j) h (hs ws n) d',
                      hs=int(self.window_size[0]*stride), ws=int(self.window_size[1]*stride), h=self.num_heads)
        return x
    
    def forward(self, x, context):
        """
        x:   [B,H,W,M,C]
        context:  [B,H*stride,W*stride,N,C]
        Returns:
            B,H,W,M,C
        """
        _, H, W, M, _ = x.shape
        N = context.shape[-2]
        q = self.projq(x)
        k, v = self.projkv(context).chunk(2, dim=-1)

        # pad feature maps to multiples of window size
        window_size = self.window_size
        pad_l = pad_t = 0
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        Hp = H + pad_b
        Wp = W + pad_r
        if pad_r > 0 or pad_b > 0:
            q = F.pad(q, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            k = F.pad(k, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
            v = F.pad(v, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))

        q = self.window_partition(q, query=True)
        k = self.window_partition(k)
        v = self.window_partition(v)

        # positional embedding
        window_size = self.window_size
        rpe = self.relative_position_enc_table[self.relative_position_index.view(-1)].view(
            window_size[0] * window_size[1], int(window_size[0] * window_size[1] * self.stride * self.stride), self.num_heads, -1)
        rpe = repeat(rpe, 'i j h c -> (i m) (j n) h c', m=M, n=N)
        q_rpe, k_rpe, v_rpe = rpe.chunk(3, dim=-1)

        # window attention
        q = q * self.scale
        q_rpe = q_rpe * self.scale
        qk = (q @ k.transpose(-2, -1))  # B head N C @ B head C N' --> B head N N'
        qr = torch.einsum('bhic,ijhc->bhij', q, k_rpe)
        kr = torch.einsum('bhjc,ijhc->bhij', k, q_rpe)
        attn = qk + qr + kr
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = attn @ v + torch.einsum('bhij,ijhc->bhic', attn, v_rpe)
        x = rearrange(x, '(b i j) h (hs ws m) d -> b (i hs) (j ws) m (h d)', 
                      i=Hp//window_size[0],
                      j=Wp//window_size[1], 
                      hs=window_size[0], 
                      ws=window_size[1])
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W].contiguous()
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    

class CrossBlock(nn.Module):
    def __init__(
        self,
        dim,
        q_dim,
        kv_dim,
        window_size,
        stride,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
    ) -> None:
        super().__init__()
        self.attn = CrossAttention(
            dim, q_dim, kv_dim, to_2tuple(window_size), stride, num_heads, qkv_bias, proj_bias, qk_scale, attn_drop, drop,
        )
        self.norm1 = norm_layer(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        
    def forward(self, x, context, x_pos=None, context_pos=None):
        shortcut = x
        if x_pos is not None:
            x = torch.cat((self.norm1(x), x_pos), dim=-1)
        else:
            x = self.norm1(x)
        if context_pos is not None:
            context = torch.cat((self.norm_y(context), context_pos), dim=-1)
        else:
            context = self.norm_y(context)
        x = shortcut + self.drop_path(self.attn(x, context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class AlignmentLayer(nn.Module):

    def __init__(
        self,
        dim,
        window_size,
        shift_size,
        num_heads,
        mlp_ratio=4.0,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        refine=False,
    ) -> None:
        super().__init__()

        kwargs = dict(
            dim=dim,
            mlp_ratio=mlp_ratio,
            window_size=window_size,
            shift_size=shift_size,
            n_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            drop_path=drop_path,
            dropout=drop,
        )
        if refine:
            self.attn = RefinementLayer(**kwargs)
        else:
            self.attn = InferenceLayer(**kwargs)

        kwargs = dict(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_mem=norm_mem,
        )
        self.readout = CrossBlock(q_dim=dim+31, kv_dim=dim, window_size=window_size, stride=0.5 if refine else 1, **kwargs)
        self.update = CrossBlock(q_dim=dim, kv_dim=dim+31, window_size=window_size // 2 if refine else window_size, stride=2 if refine else 1, **kwargs)

    def forward(self, stereo_embed, mono_embed, stereo_pos_embed, attn_mask):
        # readout
        stereo_embed = self.readout(stereo_embed, mono_embed, x_pos=stereo_pos_embed)
        # stereo aggregation
        stereo_embed = self.attn(stereo_embed, stereo_pos_embed, attn_mask)
        # update
        mono_embed = self.update(mono_embed, stereo_embed, context_pos=stereo_pos_embed)
        return stereo_embed, mono_embed


class Alignment(nn.Module):
    def __init__(
        self,
        num_blocks,
        embed_dim,
        window_size,
        num_heads,
        mlp_ratio=4.0,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        return_intermediate=False,
        refine=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.return_intermediate = return_intermediate

        dpr = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]
        self.layers = nn.ModuleList([
            AlignmentLayer(embed_dim, window_size, shift_size=0 if i % 2 == 0 else window_size // 2,
                           num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop,
                           attn_drop=attn_drop, drop_path=dpr[i], norm_layer=norm_layer, refine=refine)
            for i in range(num_blocks)
        ])

        self.norm = norm_layer(embed_dim)
        self.norm_mono = norm_layer(embed_dim)

    def compute_attn_mask(self, H, W, N, device):
        attn_mask = []
        window_size = (self.window_size, self.window_size, N)
        if N > 1:
            attn_mask.append(WindowAttention.gen_window_attn_mask(window_size, device=device)[None])
        else:
            attn_mask.append(None)

        if self.num_blocks >= 2:
            shift_size = self.window_size // 2
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            if N > 1:
                attn_mask.append(
                    WindowAttention.gen_shift_window_attn_mask((Hp, Wp), window_size, shift_size, device)
                )
            else:
                attn_mask.append(
                    gen_shift_window_attn_mask((Hp, Wp), window_size[:2], shift_size, device)
                )
        return attn_mask

    def forward(self, stereo_embed, mono_embed, stereo_pos_embed):
        """
        stereo_embed: [B,H,W,N,C]
        mono_embed: [B,H,W,C]
        sterep_pos_embed: [B,H,W,N,C]
        """
        # attention mask
        H, W, N = stereo_embed.shape[1:4]
        device = stereo_embed.device
        attn_mask = self.compute_attn_mask(H, W, N, device)

        # bidirectional alignment
        return_intermediate = self.return_intermediate and self.training
        stereo_embeds, mono_embeds = [], []
        mono_embed = mono_embed.unsqueeze(-2)
        for idx, layer in enumerate(self.layers):
            stereo_embed, mono_embed = layer(stereo_embed, mono_embed, stereo_pos_embed, attn_mask[idx%2])
            if return_intermediate:
                stereo_embeds.append(self.norm(stereo_embed))
                mono_embeds.append(self.norm_mono(mono_embed).squeeze(-2))

        if return_intermediate:
            return torch.stack(stereo_embeds, 0), torch.stack(mono_embeds, 0)
        
        return self.norm(stereo_embed)[None], self.norm_mono(mono_embed)[None].squeeze(-2)


def gen_shift_window_attn_mask(input_resolution, window_size, shift_size, device=torch.device('cuda')):
    """
    input resolution (tuple[int]):  The height and width of input
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