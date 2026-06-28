import math
import torch
import torch.nn.functional as F
from torch import nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image
from torchvision import transforms


class EA(nn.Module):
    """
    Entropy Attention Module.
    """

    def __init__(self, num_feat=64, scale_ratio=8):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat // scale_ratio, kernel_size=1)
        # calculate entropy.
        self.conv2 = nn.Conv2d(num_feat // scale_ratio, num_feat, kernel_size=1)
    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x_var = torch.var(x1, keepdim=True, dim=(2, 3))
        x_entp = torch.log(2 * math.pi * math.e * x_var) / 2.0
        entp_attn = self.conv2(x_entp)
        std_entp_attn = torch.sigmoid(entp_attn)
        return identity * std_entp_attn


class SLKA(nn.Module):
    """
    Shifting Large kernel Attention.
    """

    def __init__(self, num_feat, dw_kernel=7, dw_di_kernel=9, dw_di_dilation=3):
        super().__init__()
        dim = num_feat
        self.conv = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv3_7 = nn.Conv2d(dim // 3, dim // 3, kernel_size=3, padding=3, groups=dim // 3, dilation=3,
                                 padding_mode='reflect')
        self.conv3_13 = nn.Conv2d(dim // 3, dim // 3, kernel_size=5, padding=6, groups=dim // 3, dilation=3,
                                  padding_mode='reflect')
        self.conv3_3 = nn.Conv2d(dim // 3, dim // 3, kernel_size=3, padding=1, groups=dim // 3, dilation=1,
                                 padding_mode='reflect')
        self.conv3_19 = nn.Conv2d(dim // 3, dim // 3, kernel_size=7, padding=9, groups=dim // 3, dilation=3,
                                  padding_mode='reflect')
        self.conv3_5 = nn.Conv2d(dim // 3, dim // 3, kernel_size=5, padding=2, groups=dim // 3, dilation=1,
                                 padding_mode='reflect')
        self.conv3_21 = nn.Conv2d(dim // 3, dim // 3, kernel_size=5, padding=10, groups=dim // 3, dilation=5,
                                  padding_mode='reflect')

        self.mix1 = nn.Sequential(
            nn.Conv2d(dim // 3 * 2, 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.mix2 = nn.Sequential(
            nn.Conv2d(dim // 3 * 2, 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.mix3 = nn.Sequential(
            nn.Conv2d(dim // 3 * 2, 2, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        attention = identity
        x, y, z = attention.chunk(3, dim=1)
        x1 = self.conv3_7(x)
        x2 = self.conv3_13(x1)
        w = self.mix1(torch.cat([x1, x2], dim=1))
        x = w[:, 0, :, :].unsqueeze(1) * x1 + w[:, 1, :, :].unsqueeze(1) * x2
        y1 = self.conv3_3(y)
        y2 = self.conv3_19(y1)
        w = self.mix2(torch.cat([y1, y2], dim=1))
        y = w[:, 0, :, :].unsqueeze(1) * y1 + w[:, 1, :, :].unsqueeze(1) * y2
        z1 = self.conv3_5(z)
        z2 = self.conv3_21(z1)
        w = self.mix3(torch.cat([z1, z2], dim=1))
        z = w[:, 0, :, :].unsqueeze(1) * z1 + w[:, 1, :, :].unsqueeze(1) * z2
        out = torch.cat([x, y, z], dim=1)
        attention = self.act(self.conv(out))
        return identity * attention


class SGFN(nn.Module):

    def __init__(self, num_feat, scale=2, dw_kernel=5):
        super().__init__()
        self.sgfn_feat = num_feat * scale
        assert self.sgfn_feat % 2 == 0, "SGFN: num_feat * scale must be an even number for channel splitting."
        self.conv1 = nn.Conv2d(num_feat, self.sgfn_feat, 1)
        self.dwconv = nn.Conv2d(
            in_channels=self.sgfn_feat // 2,
            out_channels=self.sgfn_feat // 2,
            kernel_size=dw_kernel,
            padding=(dw_kernel - 1) // 2,
            groups=self.sgfn_feat // 2)
        self.conv2 = nn.Conv2d(self.sgfn_feat // 2, num_feat, 1)

    def forward(self, x):
        sgfn_feat = self.conv1(x)
        sgfn_feat_res, sgfn_feat_dw = torch.split(sgfn_feat, self.sgfn_feat // 2, dim=1)
        ffn_feat_dw = self.dwconv(sgfn_feat_dw)
        sgfn_feat = ffn_feat_dw * sgfn_feat_res
        out = self.conv2(sgfn_feat)
        return out


class EAB(nn.Module):

    def __init__(self, num_feat, scale_attn, scale_ea, scale_ffn=2, dw_kernel_ffn=5):
        super().__init__()
        # layernorm
        self.layer_norm_pre = LayerNorm2d(num_feat)

        # attention
        self.attn_feat = num_feat * scale_attn
        self.conv1 = nn.Conv2d(num_feat, self.attn_feat, 1)
        self.attn = EA(self.attn_feat, scale_ea)
        self.conv2 = nn.Conv2d(self.attn_feat, num_feat, 1)

        # LayerNorm
        self.layer_norm_aft = LayerNorm2d(num_feat)

        # SGFN
        self.ffn = SGFN(num_feat, scale_ffn, dw_kernel_ffn)

        # activation
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x
        ln_x = self.layer_norm_pre(x)

        attn_feat = self.gelu(self.conv1(ln_x))
        attn_out = self.attn(attn_feat)
        attn_out = self.conv2(attn_out) + identity

        middle_feat = attn_out
        ln_feat = self.layer_norm_aft(attn_out)

        out = self.ffn(ln_feat)

        return out + middle_feat


class MLKAB(nn.Module):

    def __init__(self,
                 num_feat,
                 scale_attn=2,
                 dw_kernel=7,
                 dw_di_kernel=9,
                 dw_di_dilation=3,
                 scale_ffn=2,
                 dw_kernel_ffn=5):
        super().__init__()
        # layernorm
        self.layer_norm_pre = LayerNorm2d(num_feat)

        # attention
        self.attn_feat = num_feat * scale_attn
        self.encoder = nn.Conv2d(num_feat, self.attn_feat, 1)
        self.slka = SLKA(self.attn_feat, dw_kernel=dw_kernel, dw_di_kernel=dw_di_kernel, dw_di_dilation=dw_di_dilation)
        self.decoder = nn.Conv2d(self.attn_feat, num_feat, 1)

        # LayerNorm
        self.layer_norm_aft = LayerNorm2d(num_feat)

        # SGFN
        self.sgfn = SGFN(num_feat, scale_ffn, dw_kernel_ffn)

        # activation
        self.gelu = nn.GELU()

    def forward(self, x):
        identity = x
        ln_x = self.layer_norm_pre(x)

        attn_feat = self.gelu(self.encoder(ln_x))
        attn_out = self.slka(attn_feat)
        attn_out = self.decoder(attn_out) + identity

        middle_feat = attn_out
        ln_feat = self.layer_norm_aft(attn_out)

        out = self.sgfn(ln_feat)

        return out + middle_feat


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x 的形状是 (B, C, H, W)
        # 在通道维度 (dim=1) 上求均值和方差，完全等价于原来的 LayerNorm
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        # 广播权重和偏置
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class DAB(nn.Module):

    def __init__(self,
                 num_feat,
                 scale_attn=2,
                 dw_kernel=7,
                 dw_di_kernel=9,
                 dw_di_dilation=3,
                 scale_ea=8,
                 scale_ffn=2,
                 dw_kernel_ffn=5):
        super().__init__()

        # spatial attention block
        self.slkab = MLKAB(
            num_feat=num_feat,
            scale_attn=scale_attn,
            dw_kernel=dw_kernel,
            dw_di_kernel=dw_di_kernel,
            dw_di_dilation=dw_di_dilation,
            scale_ffn=scale_ffn,
            dw_kernel_ffn=dw_kernel_ffn)

        # channel attention block
        self.eab = EAB(
            num_feat=num_feat,
            scale_attn=scale_attn,
            scale_ea=scale_ea,
            scale_ffn=scale_ffn,
            dw_kernel_ffn=dw_kernel_ffn)

    def forward(self, x):
        return self.eab(self.slkab(x))


class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor**2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class PixelShuffleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(in_channels, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample(upscale_factor, num_feat)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


#@ARCH_REGISTRY.register()
class HFENet(nn.Module):

    def __init__(self,
                 num_feat=27,
                 num_blocks=7,
                 scale_attn=2,
                 dw_kernel=5,
                 dw_di_kernel=7,
                 dw_di_dilation=3,
                 scale_ea=8,
                 scale_ffn=2,
                 dw_kernel_ffn=3,
                 upsampler='pixelshuffledirect',
                 img_range=1.,
                 upscale=4):
        super().__init__()
        self.img_range = img_range
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        # shallow feature process
        self.simple_encoder = nn.Conv2d(3, num_feat, 3, padding=1)

        # deep feature process
        self.pipeline = nn.Sequential(*[
            DAB(num_feat=num_feat,
                scale_attn=scale_attn,
                dw_kernel=dw_kernel,
                dw_di_kernel=dw_di_kernel,
                dw_di_dilation=dw_di_dilation,
                scale_ea=scale_ea,
                scale_ffn=scale_ffn,
                dw_kernel_ffn=dw_kernel_ffn) for _ in range(num_blocks)
        ])

        self.after_pipe = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.gelu = nn.GELU()

        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep(num_feat, 3, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle':
            self.upsampler = PixelShuffleBlock(num_feat, 3, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=True):
            self.mean = self.mean.type_as(x)
            x = (x - self.mean) * self.img_range
            out_fea = self.gelu(self.simple_encoder(x))
            pipeline_out = self.after_pipe(self.pipeline(out_fea)) + out_fea
            out = self.upsampler(pipeline_out)
            out = out / self.img_range + self.mean
        return out.float()
