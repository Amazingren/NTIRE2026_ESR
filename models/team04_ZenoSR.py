import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

# from basicsr.utils.registry import ARCH_REGISTRY


# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


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


# Attention
class Attention(nn.Module):
    def __init__(self, dim, tlc_flag=True, tlc_kernel=72):
        super(Attention, self).__init__()

        self.tlc_flag = tlc_flag    # TLC flag for validation and test

        self.temperature = nn.Parameter(torch.ones(1, 1))

        self.project_in = nn.Conv2d(dim, dim * 4, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.act = nn.Softmax(dim=-1)

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qv):
        q, v = qv.chunk(2, dim=1)

        q = rearrange(q, 'b c h w -> b c (h w)')
        v = rearrange(v, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(v, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = self.act(attn)

        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        x1, x2 = self.project_in(x).chunk(2, dim=1)

        qv = self.dwconv(x1) * x2

        if self.training or not self.tlc_flag:
            out = self._forward(qv)
            out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qv = self.grids(qv)  # convert to local windows
        out = self._forward(qv)
        out = rearrange(out, 'b c (h w) -> b c h w', h=qv.shape[-2], w=qv.shape[-1])
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)

        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 2, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt


class ChannelAttnBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=72):
        super(ChannelAttnBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.global_channel = Attention(dim, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel)
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=False)

    def forward(self, x):
        x = self.global_channel(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

class SpatialAttnBlock(nn.Module):
    def __init__(self, dim):
        super(SpatialAttnBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // 2, 1, 1, 0, bias=False)
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, 3, 1, 1, groups=dim // 2, bias=False)
        self.conv2 = nn.Conv2d(dim // 2, dim // 2, 1, 1, 0, bias=False)
        self.conv3 = nn.Conv2d(dim // 2, dim, 1, 1, 0, bias=False)

    def forward(self, x):
        _,_,h,w = x.shape
        x0 = F.normalize(x)
        x1 = self.conv1(x0)
        x2 = self.dwconv(F.adaptive_max_pool2d(x1, (h//8, w//8)))
        x3 = torch.var(x1, dim=(-2,-1), keepdim=True)
        x4 = F.interpolate(F.gelu(self.conv2(x2 + x3)), size=(h,w), mode='nearest')
        out = self.conv3(x1 * x4)
        return out + x

class LocalHandleBlock(nn.Module):
    def __init__(self, dim):
        super(LocalHandleBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1, 1, 0, bias=True)
        self.conv3 = nn.Conv2d(dim, dim // 2, 1, 1, 0, bias=True)
        self.conv4 = nn.Conv2d(dim, dim // 2, 3, 1, 1, groups=dim // 2, bias=True)
        self.dwconv1 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0, bias=True),
                                     nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True))
        self.dwconv2 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0, bias=True),
                                     nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True))
        self.dwconv3 = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0, bias=True),
                                     nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True))
        self.conv5 = nn.Conv2d(dim * 2, dim, 1, 1, 0, bias=True)

        # self.alpha_1 = nn.Parameter(torch.ones((1, dim // 2, 1, 1)), requires_grad=True)
        # self.alpha_2 = nn.Parameter(torch.ones((1, dim // 2, 1, 1)), requires_grad=True)
        # self.alpha_3 = nn.Parameter(torch.ones((1, dim // 2, 1, 1)), requires_grad=True)
        # self.alpha_4 = nn.Parameter(torch.ones((1, dim // 2, 1, 1)), requires_grad=True)

    def forward(self, x):
        x1 = F.gelu(self.dwconv1(x) + x)
        x2 = F.gelu(self.dwconv2(x1) + x1)
        x3 = F.gelu(self.dwconv3(x2) + x2)
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(x1))
        out3 = F.gelu(self.conv3(x2))
        out4 = F.gelu(self.conv4(x3))
        # out = torch.cat([self.alpha_1 * out1, self.alpha_2 * out2, self.alpha_3 * out3, self.alpha_4 * out4], dim=1)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.conv5(out)
        return out

class SCAB(nn.Module):
    def __init__(self, dim):
        super(SCAB, self).__init__()

        self.local_handle = LocalHandleBlock(dim)
        self.spatial_attn = SpatialAttnBlock(dim)
        self.channel_attn = ChannelAttnBlock(dim)

    def forward(self, x):
        out = self.local_handle(x)
        out = self.spatial_attn(out)
        out = self.channel_attn(out)
        return out + x

"""
class DAN(nn.Module):
    def __init__(self, dim=16, upscale=4):
        super().__init__()
"""


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, padding=0, stride=s, bias=bias)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in * gain, kernel_size=1, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_in * gain, out_channels=c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
            nn.Conv2d(in_channels=c_out * gain, out_channels=c_out, kernel_size=1, padding=0, bias=bias),
        )

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.data.clone().detach()
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.data.clone().detach()
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


# @ARCH_REGISTRY.register()
class ZenoSR(nn.Module):
    def __init__(self, dim = 18, upscale = 4):
        super(ZenoSR, self).__init__()

        # MeanShift for Image Input
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=False)
        # self.to_feat = Conv3XC(3, dim, gain1=2, s=1)

        self.B1 = SCAB(dim)
        self.B2 = SCAB(dim)
        self.B3 = SCAB(dim)
        self.B4 = SCAB(dim)
        # self.B5 = SCAB(dim)
        # self.B6 = SCAB(dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim*4, dim, 1, 1, 0, bias=False),
            nn.GELU(),
            LocalHandleBlock(dim)
        )

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale**2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale)
        )

        modules = []
        for _ in range(4):
            modules.append(nn.GELU())
            modules.append(nn.Conv2d(3, 3, 3, 1, 1, groups=3, bias=False))

        self.refine = nn.Sequential(*modules)

        self.alpha_5 = nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.alpha_6 = nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.alpha_7 = nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.alpha_8 = nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = x - self.mean

        x0 = self.to_feat(x)

        x1 = self.B1(x0)
        x2 = self.B2(x1)
        x3 = self.B3(x2)
        x4 = self.B4(x3)
        # x5 = self.B5(x4)
        # x6 = self.B6(x5)
        
        # out = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        out = torch.cat([self.alpha_5 * x1, self.alpha_6 * x2, self.alpha_7 * x3, self.alpha_8 * x4], dim=1)

        out = self.fuse(out) + x0
        out = self.to_img(out)
        out = self.refine(out)

        out = out + self.mean
        return out



        
        


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    x = torch.randn(1, 3, 256, 256)

    model = DAN(dim=16, upscale=4)

    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)