#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @File    : net.py
# @Software: vscode
# @Desc    : description
# @Author  : Shaowu wu
# @Email    : wshaowu@whu.edu.cn

from functools import partial
import math
import numbers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import init
from tqdm import tqdm

from networks.base_plmodel import BaseModel

##########################################################################################################################
# ------Rformer Architecture--------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class TransformerBlock_onlyFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type):
        super().__init__()
        # self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # x = x + self.ffn(self.norm2(x))
        x = x + self.ffn(x)
        return x


# 借鉴restore former的结构但是都没有下采样操作
class EncoderRformer(nn.Module):
    def __init__(self, ms_chans, dim, ffn_expansion_factor=2.66, num_blocks=1):
        super().__init__()
        self.ms_embed = OverlapPatchEmbed(ms_chans, dim)
        self.pan_embed = OverlapPatchEmbed(1, dim)
        self.noise_ms_embed = OverlapPatchEmbed(ms_chans, dim)

        self.ms_body = nn.Sequential(*[TransformerBlock_onlyFFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias') for _ in range(num_blocks)])
        self.pan_body = nn.Sequential(*[TransformerBlock_onlyFFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias') for _ in range(num_blocks)])
        self.noise_ms_body = nn.Sequential(*[TransformerBlock_onlyFFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type='WithBias') for _ in range(num_blocks)])

    def forward(self, ms, pan, noise_upms=None):
        ms = self.ms_embed(ms)
        ms = self.ms_body(ms)
        pan = self.pan_embed(pan)
        pan = self.pan_body(pan)
        if noise_upms is not None:
            noise_upms = self.noise_ms_embed(noise_upms)
            noise_upms = self.noise_ms_body(noise_upms)
        return ms, pan, noise_upms


##########################################################################################################################
## ------EDSR Architecture--------
class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size // 2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
# 借鉴EDSR的结构但是都没有下采样操作
class EncoderEDSR(nn.Module):
    def __init__(self, ms_chans, n_feats, n_resblocks=1):
        super().__init__()
        self.ms_head = nn.Conv2d(ms_chans, n_feats, 3, padding=1)
        self.pan_head = nn.Conv2d(1, n_feats, 3, padding=1)
        self.xt_head = nn.Conv2d(ms_chans, n_feats, 3, padding=1)
        ms_body = [ResBlock(n_feats, 3) for _ in range(n_resblocks)]
        pan_body = [ResBlock(n_feats, 3) for _ in range(n_resblocks)]

        xt_body = [ResBlock(n_feats, 3) for _ in range(n_resblocks)]

        self.ms_body = nn.Sequential(*ms_body)
        self.pan_body = nn.Sequential(*pan_body)
        self.xt_body = nn.Sequential(*xt_body)
        
    def forward(self, ms, pan, noise_upms=None, timeemd=None):
        # ms: [b, c, h, w], pan: [b, c, h, w], noise_upms: [b, c, h, w]
        ms = self.ms_head(ms)
        ms_feats = self.ms_body(ms)

        pan = self.pan_head(pan)
        pan_feats = self.pan_body(pan)

        if noise_upms is not None:
            xt = self.xt_head(noise_upms)
            xt_feats = self.xt_body(xt)
        else:
            xt_feats = None

        return ms_feats, pan_feats, xt_feats
##########################################################################################################################


##########################################################################################################################
class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*(1+self.use_affine_level), kernel_size=1, padding=0, bias=True)
        )

    def forward(self, x, noise_embed):
        if noise_embed is None:
            return 0
        batch = x.shape[0] # B C H W 
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed.view(
                batch, -1, 1, 1)).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed.view(batch, -1, 1, 1))
        return x

class FusionResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, cat_feats=False, use_affine_level=False, is_first=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.cat_feats = cat_feats
        if cat_feats:
            self.cat_conv = nn.Conv2d(n_feats*2 if is_first else n_feats*3, n_feats, kernel_size=kernel_size, padding=kernel_size // 2)
        else:
            self.cat_conv = nn.Conv2d(n_feats, n_feats, kernel_size=kernel_size, padding=kernel_size // 2)

        self.time_inject = FeatureWiseAffine(n_feats, n_feats, use_affine_level=False)
        self.ms_pan_resblock = ResBlock(n_feats, kernel_size)
        self.activation = nn.ReLU()


    def forward(self, ms_feats, pan_feats, xt_feats, noise_embed, shape):
        
        if self.cat_feats:
            if xt_feats is not None:
                inp = torch.cat([ms_feats, pan_feats, xt_feats], dim=-1)
            else:
                inp = torch.cat([ms_feats, pan_feats], dim=-1)
        else:
            if xt_feats is not None:
                inp = ms_feats + pan_feats + xt_feats
            else:
                inp = ms_feats + pan_feats
        B, C, H, W = shape
        inp = rearrange(inp, "b (h w) c -> b c h w", h=H, w=W)
        fused_feat = self.cat_conv(inp)
        fused_feat = self.activation(fused_feat)
        fused_feat = self.time_inject(fused_feat, noise_embed) + fused_feat
        fused_feat = self.ms_pan_resblock(fused_feat)
        fused_feat = rearrange(fused_feat, "b c h w -> b (h w) c")
        return fused_feat

def make_coord(shape:tuple, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def compute_hi_coord(coord, n):
    coord_clip = torch.clip(coord - 1e-9, 0., 1.)
    coord_bin = ((coord_clip * 2 ** (n + 1)).floor() % 2)
    return coord_bin

class MLP_with_shortcut(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        short_cut = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if x.shape[-1] == short_cut.shape[-1]:
            x = x + short_cut
        return x

class Decoder(nn.Module):
    def __init__(self, n_feats, n_layers=5, use_cell=True, use_coord_code=False, dense_block=False):
        super().__init__()
        self.use_coord_code = use_coord_code
        self.n_layers = n_layers
        self.use_cell = use_cell
        self.ms_attn = TransformerBlock_onlyFFN(n_feats, 2.66, False, 'WithBias') # 4 is the number of heads
        self.pan_attn = TransformerBlock_onlyFFN(n_feats, 2.66, False, 'WithBias')
        self.xt_attn = TransformerBlock_onlyFFN(n_feats, 2.66, False, 'WithBias')
        self.ms_project_in = nn.Conv2d(n_feats*4, n_feats, kernel_size=1, padding=0)
        self.pan_project_in = nn.Conv2d(n_feats*4, n_feats, kernel_size=1, padding=0)
        self.xt_project_in = nn.Conv2d(n_feats*4, n_feats, kernel_size=1, padding=0)
        self.ms_fc_layers = nn.ModuleList([MLP_with_shortcut(n_feats + 4 if d==0 else n_feats , n_feats, 2*n_feats) for d in range(n_layers)])
        self.xt_fc_layer = MLP_with_shortcut(n_feats + 4, n_feats, 2*n_feats)
        # self.pan_fc_layers = nn.ModuleList([nn.Linear(n_feats + 2 if d == 0 else n_feats , n_feats, 2*n_feats) for d in range(n_layers)])
        self.pan_fc_layers = nn.ModuleList([MLP_with_shortcut(n_feats + 4 if d==0 else n_feats , n_feats, 2*n_feats) for d in range(n_layers)])
        if dense_block:
            self.funse_layers = nn.ModuleList([FusionResBlock(n_feats) for _ in range(n_layers)])
        else:
            self.funse_layers = nn.ModuleList([FusionResBlock(n_feats) for _ in range(n_layers)])
    def query_coord_feat(self, coord, feats: torch.Tensor):
        pos_feats = make_coord(feats.shape[-2:], flatten=False).to(feats.device).permute(2, 0, 1).unsqueeze(0).expand(feats.shape[0], 2, *feats.shape[-2:])
        rx = 2 / feats.shape[-2] / 2
        ry = 2 / feats.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feats, coord_.flip(-1), mode='nearest', align_corners=False)

                old_coord = F.grid_sample(pos_feats, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feats.shape[-2] / 2
                rel_coord[:, 1, :, :] *= feats.shape[-1] / 2
                rel_coord_n = rel_coord.permute(0, 2, 3, 1).reshape(rel_coord.shape[0], -1, rel_coord.shape[1]) # B H W 2 -> B HW 2

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                preds.append(feat_)
                if vx == -1 and vy == -1:
                    rel_coord_mask = (rel_coord_n > 0).float() # B HW 2
                    rxry = torch.tensor([rx, ry], device=coord.device)[None, None, :] 
                    local_coord = rel_coord_mask * rel_coord_n + (1. - rel_coord_mask) * (rxry - rel_coord_n)
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t
        for index, area in enumerate(areas):
            preds[index] = preds[index] * (area / tot_area).unsqueeze(1)
        preds = torch.cat(preds, dim=1)
        rel_coord = rearrange(rel_coord, "b c h w -> b (h w) c")
        return preds, local_coord, rel_coord
        
    def forward(self, ms_feats, pan_feats, xt_feats, coord=None, noise_embed=None):
        # ms_feats, pan_feats, xt_feats: [B, C, H, W], coord: [B, 2, H, W], noise_embed: [B, C]
        ms_feats = self.ms_attn(ms_feats) # [B, C, H, W]
        pan_feats = self.pan_attn(pan_feats) # [B, C, H, W]
        

        pan_feats, pan_coord, pan_rel_coord = self.query_coord_feat(coord, pan_feats)
        ms_feats, ms_coord, ms_rel_coord = self.query_coord_feat(coord, ms_feats)

        pan_feats = self.pan_project_in(pan_feats)
        ms_feats = self.ms_project_in(ms_feats)

        coord_shape = ms_feats.shape
        ms_feats = rearrange(ms_feats, 'b c h w -> b (h w) c')
        pan_feats = rearrange(pan_feats, 'b c h w -> b (h w) c')

        if xt_feats is not None:
            xt_feats = self.xt_attn(xt_feats) # [B, C, H, W]
            xt_feats, xt_coord, xt_rel_coord = self.query_coord_feat(coord, xt_feats)
            xt_feats = self.xt_project_in(xt_feats)
            xt_feats = rearrange(xt_feats, 'b c h w -> b (h w) c')

        for i in range(self.n_layers):
            if i == 0:
                if self.use_coord_code:
                    ms_input = torch.cat([ms_feats, ms_coord], dim=-1)
                    pan_input = torch.cat([pan_feats, pan_coord], dim=-1)
                    xt_feats = torch.cat([xt_feats, xt_coord], dim=-1) if xt_feats is not None else None
                else:
                    ms_input = torch.cat([ms_feats, ms_rel_coord], dim=-1)
                    pan_input = torch.cat([pan_feats, pan_rel_coord], dim=-1)
                    xt_feats = torch.cat([xt_feats, xt_rel_coord], dim=-1) if xt_feats is not None else None
                if self.use_cell:
                    cell = torch.ones([coord_shape[0], coord_shape[-2]*coord_shape[-1], 2]).to(device=ms_feats.device) # (b, h, w, 2)
                    cell[:, :, 0] *= 2 / coord.shape[-2] # 因为是正负1，所以是2
                    cell[:, :, 1] *= 2 / coord.shape[-1]
                    ms_rel_cell = cell.clone()
                    ms_rel_cell[:, :, 0] *= ms_feats.shape[-2]
                    ms_rel_cell[:, :, 1] *= ms_feats.shape[-1]
                    ms_input = torch.cat([ms_input, ms_rel_cell], dim=-1)

                    pan_rel_cell = cell.clone()
                    pan_rel_cell[:, :, 0] *= pan_feats.shape[-2]
                    pan_rel_cell[:, :, 1] *= pan_feats.shape[-1]
                    pan_input = torch.cat([pan_input, pan_rel_cell], dim=-1)

                    if xt_feats is not None:
                        xt_rel_cell = cell.clone()
                        xt_rel_cell[:, :, 0] *= xt_feats.shape[-2]
                        xt_rel_cell[:, :, 1] *= xt_feats.shape[-1]
                        xt_feats = torch.cat([xt_feats, xt_rel_cell], dim=-1)
                        xt_feats = self.xt_fc_layer(xt_feats)
            else:
                ms_input = ms_feats
                pan_input = pan_feats

            ms_feats = self.ms_fc_layers[i](ms_input)
            pan_feats = self.pan_fc_layers[i](pan_input)
            xt_feats = self.funse_layers[i](ms_feats, pan_feats, xt_feats, noise_embed, coord_shape) # 这里可以考虑只在第一个位置加入noise embed
        
        xt_feats = rearrange(xt_feats, 'b (h w) c -> b c h w', h=coord_shape[-2], w=coord_shape[-1])
            
        return xt_feats # B C H W

##########################################################################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)
    
class UnetModel(nn.Module):
    def __init__(self, ms_chans, en_dim, encoder_type='rformer'):
        super().__init__()
        if encoder_type == 'edsr':
            self.encoder = EncoderEDSR(ms_chans, en_dim)
        elif encoder_type == 'rformer':
            self.encoder = EncoderRformer(ms_chans, en_dim)
        else:
            raise NotImplementedError('encoder_type: {}'.format(encoder_type))
        self.decoder = Decoder(en_dim, en_dim)
        
        self.head = nn.Conv2d(en_dim, ms_chans, 1, padding=0)

        self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(en_dim),
                nn.Linear(en_dim, en_dim * 2),
                Swish(),
                nn.Linear(en_dim * 2, en_dim)
            )
    def forward(self, ms, pan, coord=None, noise_upms=None, t=None):
        if t is not None:
            time_emd = self.noise_level_mlp(torch.tensor([t]).to(ms.device).expand(ms.shape[0], 1))
        else:
            time_emd = None
        ms_feats, pan_feats, xt_feats = self.encoder(ms, pan, noise_upms)
        ms_pan_feats = self.decoder(ms_feats, pan_feats, xt_feats,coord, time_emd)
        out = self.head(ms_pan_feats)
        ms_up = F.interpolate(ms, size=coord.shape[1:3], mode='bilinear', align_corners=False)
        return out + ms_up

def make_beta_schedule(n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, schedule='linear'):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

class DDPMDiffusion(nn.Module):
    def __init__(
            self,
            ms_chans=4,
            en_dim=32,
            resolution_step=1,
            min_img_size = 64,
            max_img_size = 256,

            # diffusion
            n_ddpm_time_steps=2000,
            linear_start = 1e-6,
            linear_end = 1e-2,

    ):
        super().__init__()
        self.model = UnetModel(ms_chans=ms_chans, en_dim=en_dim)
        self.gamma = 200
        self.min_img_size = min_img_size
        self.max_img_size = max_img_size
        self.resolution_step = resolution_step
        self.resolution_set = np.arange(max_img_size, min_img_size-1-resolution_step, -resolution_step)
        if self.resolution_set[-1] < min_img_size:
            self.resolution_set[-1] = min_img_size
        self.n_time_steps = self.resolution_set.shape[0]

        self.n_ddpm_time_steps = n_ddpm_time_steps
        self.ddpm_step = self.n_ddpm_time_steps // self.n_time_steps
        self.ddpm_time_steps_set = np.arange(1, self.n_ddpm_time_steps+self.ddpm_step, self.ddpm_step)
        if self.ddpm_time_steps_set[-1] > self.n_ddpm_time_steps:
            self.ddpm_time_steps_set[-1] = self.n_ddpm_time_steps
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.set_noise_schedule()
        self.loss = nn.L1Loss()
    
    def set_noise_schedule(self,):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = make_beta_schedule(self.n_ddpm_time_steps, self.linear_start, self.linear_end)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.register_buffer('ddim_c1', torch.sqrt(to_torch((1. - alphas_cumprod / alphas_cumprod_prev) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod))))

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )
    @torch.no_grad()
    def pansharpening(self, ms, pan):
        b = ms.shape[0]
        noise = torch.randn_like(ms)
        xt = ms
        for t in tqdm(reversed(range(1, self.n_time_steps)), desc='sampling loop time step', total=self.n_time_steps-1):
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[self.ddpm_time_steps_set[t-1]],
                self.sqrt_alphas_cumprod_prev[self.ddpm_time_steps_set[t]],
                size=b
                )
            ).to(ms.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)
            continuous_sqrt_alpha_cumprod = torch.clip(continuous_sqrt_alpha_cumprod, 0.5, 1)

            # 使用离散alpha
            # continuous_sqrt_alpha_cumprod = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[self.ddpm_time_steps_set[t]]]).to(ms.device).expand(b, 1)
            # continuous_sqrt_alpha_cumprod = torch.FloatTensor([1.0]).to(ms.device).expand(b, 1)
            

            noise = torch.randn_like(xt)
            x_noisy = self.q_sample(x_start=xt, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
            coord = make_coord((self.resolution_set[t-1], self.resolution_set[t-1]), flatten=False)
            coord = coord.to(ms.device).unsqueeze(0).expand(ms.shape[0], self.resolution_set[t-1], self.resolution_set[t-1], 2)

            xt = self.model(ms, pan, coord, x_noisy, t)
        return xt
            

    def p_losses(self, ms, pan, gt, t):
        b = ms.shape[0]
        xt_size = self.resolution_set[t]
        xt_minus1_size = self.resolution_set[t-1]

        xt = F.interpolate(gt, size=(xt_size, xt_size), mode='bilinear', align_corners=False)
        xt_minus1 = F.interpolate(gt, size=(xt_minus1_size, xt_minus1_size), mode='bilinear', align_corners=False)

        noise = torch.randn_like(xt)

        # 这里考虑使用离散的
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[self.ddpm_time_steps_set[t-1]],
                self.sqrt_alphas_cumprod_prev[self.ddpm_time_steps_set[t]],
                size=b
            )
        ).to(xt.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1) 
        continuous_sqrt_alpha_cumprod = torch.clip(continuous_sqrt_alpha_cumprod, 0.5, 1)

        # 使用离散alpha
        # continuous_sqrt_alpha_cumprod = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[self.ddpm_time_steps_set[t]]]).to(ms.device).expand(b, 1)
        # no noise
        # continuous_sqrt_alpha_cumprod = torch.FloatTensor([1.0]).to(ms.device).expand(b, 1)

        x_noisy = self.q_sample(
            x_start=xt, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)
        
        coord = make_coord((xt_minus1_size, xt_minus1_size), flatten=False)
        coord = coord.to(xt.device).unsqueeze(0).expand(xt.shape[0], *xt_minus1.shape[-2:], 2)
        pred_xt_minus1 = self.model(ms, pan, coord, x_noisy, t)
        loss = self.loss(pred_xt_minus1, xt_minus1)
        
        return loss

    def forward(self, ms, pan, gt):
        # 随机采样时间
        t = np.random.randint(1, self.n_time_steps)
        return self.p_losses(ms, pan, gt, t)

class Net(BaseModel):
    def __init__(self, ms_chans, dim, resolution_step, isFR=False, need_interpolate=False):
        super().__init__()
        self.isFR = isFR
        self.need_interpolate = need_interpolate
        self.diffusion_model = DDPMDiffusion(ms_chans=ms_chans, en_dim=dim, resolution_step=resolution_step)
        # self.net = UnetModel(ms_chans=ms_chans,en_dim=dim)
        self.criterion = nn.L1Loss()

    def forward(self, batch):
        lms, pan, wave_lengths = batch['LR'], batch['REF'], batch['wave_length']
        if self.need_interpolate:
            lms = torch.nn.functional.interpolate(lms, size=pan.shape[-2:], mode='bicubic',
                                                  align_corners=False)
        if self.current_phase == 'train':
            loss = self.diffusion_model(lms, pan, batch['GT'])
            sr = None
        else:
            sr = self.diffusion_model.pansharpening(lms, pan)
            loss = None
        return {'sr': sr, 'loss': loss}

    def configure_optimizers(self):
        lr = self.learning_rate
        opt = torch.optim.AdamW(self.diffusion_model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.02, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=500, gamma=0.99)
        return [opt], [scheduler]


###########################################################################################################################
# class Net(BaseModel):
#     def __init__(self, ms_chans, dim, isFR=False, need_interpolate=False):
#         super().__init__()
#         self.isFR = isFR
#         self.need_interpolate = need_interpolate
#         self.net = UnetModel(ms_chans=ms_chans,en_dim=dim)
#         self.criterion = nn.L1Loss()

#     def forward(self, batch):
#         lms, pan, wave_lengths = batch['LR'], batch['REF'], batch['wave_length']
#         if self.need_interpolate:
#             lms = torch.nn.functional.interpolate(lms, size=pan.shape[-2:], mode='bicubic',
#                                                   align_corners=False)
#         if self.current_phase == 'train':
#             random_size = np.random.randint(low=lms.shape[-1], high=pan.shape[-1])
#             coord = make_coord((random_size, random_size), flatten=False)
#             coord = coord.to(lms.device).unsqueeze(0).expand(lms.shape[0], random_size, random_size, 2)
#             batch['GT'] = torch.nn.functional.interpolate(batch['GT'], size=(random_size, random_size), mode='bilinear',)
#         else:
#             coord = make_coord((pan.shape[-2], pan.shape[-1]), flatten=False)
#             coord = coord.to(lms.device).unsqueeze(0).expand(lms.shape[0], pan.shape[-2], pan.shape[-1], 2)
        
#         sr = self.net(lms, pan, coord=coord)

#         return {'sr': sr, 'loss': None}

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         opt = torch.optim.AdamW(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.02, eps=1e-8)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt, step_size=300, gamma=0.99)
#         return [opt], [scheduler]


if __name__ == '__main__':
    ms = torch.rand(1, 4, 64, 64)
    pan = torch.rand(1, 1, 256, 256)
    gt = torch.rand(1, 4, 256, 256)
    # noise_up = torch.rand(1, 4, 222, 222)
    # coord = make_coord((256, 256), flatten=False)
    # coord = coord.to(ms.device).unsqueeze(0).expand(ms.shape[0], 256, 256, 2)
    # t = torch.tensor([222 - 64])

    # # test EncoderEDSR
    # e = EncoderEDSR(4, 64)
    # ms_, pan_, xt = e(ms, pan, noise_up)
    # print(ms_.shape, pan_.shape, xt.shape)

    # # test EncoderRformer
    # e = EncoderRformer(4, 64)
    # ms_, pan_, xt = e(ms, pan, noise_up)
    # print(ms_.shape, pan_.shape, xt.shape)


    # test Decoder
    # d = Decoder(64, 3)
    # ms_feats = torch.randn(1, 64, 256, 256)
    # pan_feats = torch.randn(1, 64, 256, 256)
    # xt_feats = torch.randn(1, 64, 125, 125)
    # coord = make_coord((125, 125), flatten=False)
    # coord = coord.to(xt_feats.device).unsqueeze(0).expand(xt_feats.shape[0], *xt_feats.shape[-2:], 2)
    # noise_embed = torch.randn(1, 64)
    # out = d(ms_feats, pan_feats, xt_feats, coord, noise_embed)
    # print(out.shape)

    # test Unet
    # unet = UnetModel(ms_chans=4, en_dim=64)
    # out = unet(ms, pan, coord=coord)
    # print(out.shape)

    # test DDPMDiffusion
    d = DDPMDiffusion(ms_chans=4, en_dim=64)
    out = d(ms, pan, gt)
    print(out.shape)
    p = d.pansharpening(ms, pan)
    print(p.shape)









