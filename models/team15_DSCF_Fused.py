from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)

class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.has_relu = relu
        self.eval_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out,
            kernel_size=3, padding=1, stride=s, bias=bias
        )

    def forward(self, x):
        out = self.eval_conv(x)
        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out 

class SPAB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)
        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)
        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3).sub_(0.5)
        out = out3.add(x).mul_(sim_att)
        return out, out1, out2, out3

class SPAB_Fused(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB_Fused, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, out_feature, out1_fused):
        out1_act = self.act1(out1_fused)
        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)
        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3).sub_(0.5)
        out = out3.add(out_feature).mul_(sim_att)
        return out, out1_fused, out2, out3

class DSCF_Fused(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=26,
                 upscale=4,
                 bias=True,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)
                 ):
        super(DSCF_Fused, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.conv_fused = nn.Conv2d(in_channels, feature_channels, kernel_size=5, padding=2, bias=True)
        
        self.block_1 = SPAB_Fused(feature_channels, bias=bias)
        self.block_2 = SPAB(feature_channels, bias=bias)
        self.block_3 = SPAB(feature_channels, bias=bias)
        self.block_4 = SPAB(feature_channels, bias=bias)
        self.block_5 = SPAB(feature_channels, bias=bias)
        self.block_6 = SPAB(feature_channels, bias=bias)

        self.conv_cat_other = nn.Conv2d(feature_channels * 3, feature_channels, kernel_size=1, bias=True)
        self.conv2_fused = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False)
        
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)
            
    def forward(self, x):
        if getattr(self, 'mean_allocated', None) is None or self.mean.device != x.device or self.mean.dtype != x.dtype:
            self.mean = self.mean.to(device=x.device, dtype=x.dtype)
            self.mean_allocated = True
            
        x_norm = x.sub(self.mean).mul_(self.img_range)

        out_feature = self.conv_1(x_norm)
        out1_fused = self.conv_fused(x_norm)

        out_b1, _, _, _ = self.block_1(out_feature, out1_fused)
        
        out_b2, _, _, _ = self.block_2(out_b1)
        out_b3, _, _, _ = self.block_3(out_b2)
        out_b4, _, _, _ = self.block_4(out_b3)
        out_b5, _, _, _ = self.block_5(out_b4)
        out_b6, out_b6_1, _, _ = self.block_6(out_b5)

        other_features = torch.cat([out_feature, out_b1, out_b6_1], 1)
        out_other = self.conv_cat_other(other_features)
        
        out_b6_branch = self.conv2_fused(out_b6)
        
        out = out_other + out_b6_branch
        
        output = self.upsampler(out)

        return output
