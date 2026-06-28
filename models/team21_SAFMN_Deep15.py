"""
SAFMN_Deep15 - Spatially-Adaptive Feature Modulation Network (Deep 15-Block Variant)
NTIRE 2026 Efficient Super-Resolution Challenge - Team 21

Architecture: SAFM blocks with multi-scale spatial feature modulation.
- dim=40, num_blocks=15, ffn_scale=2.0
- Parameters: 0.149M (149,248)
- FLOPs: 9.62G (on 256x256 input)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
    def forward(self, x):
        return self.norm(x)

class SAFM(nn.Module):
    """Spatially-Adaptive Feature Modulation"""
    def __init__(self, dim, n_levels=4):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList([
            nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)
            for _ in range(self.n_levels)
        ])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (max(1, h // (2 ** i)), max(1, w // (2 ** i)))
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out

class SAFM_Block(nn.Module):
    """SAFM Block with Feed-Forward Network"""
    def __init__(self, dim, ffn_scale=2.0):
        super().__init__()
        self.norm1 = ChannelNorm(dim)
        self.safm = SAFM(dim)
        self.norm2 = ChannelNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * ffn_scale), 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(int(dim * ffn_scale), dim, 1, 1, 0)
        )

    def forward(self, x):
        y = self.safm(self.norm1(x)) + x
        out = self.ffn(self.norm2(y)) + y
        return out

class SAFMN_Deep15(nn.Module):
    """SAFMN with 15 deep blocks for efficient super-resolution."""
    def __init__(self, num_in_ch=3, num_out_ch=3, dim=40, num_blocks=15, upscale=4):
        super().__init__()
        self.to_feat = nn.Conv2d(num_in_ch, dim, 3, 1, 1)
        self.blocks = nn.Sequential(*[SAFM_Block(dim) for _ in range(num_blocks)])
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, num_out_ch * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale)
        )
        self.upscale = upscale

    def forward(self, x):
        feat = self.to_feat(x)
        feat = self.blocks(feat) + feat
        out = self.upsampler(feat)
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear',
                             align_corners=False)
        return out + base
