import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

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


class GDFN(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

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


# Gated Self-Attention
class GSA(nn.Module):
    def __init__(self, dim, tlc_flag=True, tlc_kernel=72):
        super(GSA, self).__init__()

        self.tlc_flag = tlc_flag  # TLC flag for validation and test

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

# Global Channel Attention Block
class GCAB(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=72):
        super(GCAB, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.global_channel = GSA(dim, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel)
        self.ffn = GDFN(dim, ffn_expansion_factor=ffn_expansion_factor, bias=False)

    def forward(self, x):
        x = self.global_channel(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x

# Variance-guided Contextual Spatial Attention
class VCSA(nn.Module):
    def __init__(self, dim):
        super(VCSA, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim // 2, 1, 1, 0, bias=False)
        self.dwconv = nn.Conv2d(dim // 2, dim // 2, 3, 1, 1, groups=dim // 2, bias=False)
        self.conv2 = nn.Conv2d(dim // 2, dim // 2, 1, 1, 0, bias=False)
        self.conv3 = nn.Conv2d(dim//2, dim, 1, 1, 0, bias=False)
        self.alpha = nn.Parameter(torch.ones((1, dim//2, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim//2, 1, 1)))

    def forward(self, x):
        _, _, h, w = x.shape
        x0 = F.normalize(x)
        x1 = self.conv1(x0)
        x2 = self.dwconv(F.adaptive_max_pool2d(x1, (h // 8, w // 8)))
        x3 = torch.var(x1, dim=(-2, -1), keepdim=True)
        x4 = F.interpolate(F.gelu(self.conv2(self.alpha * x2 + self.belt * x3)), size=(h, w), mode='nearest')
        out = self.conv3(x1 * x4)
        return out + x

# Multi-scale Convolution
class MSConv(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dim = in_dim // 4
        self.conv1 = nn.Conv2d(in_dim, in_dim, 1,1,0,bias=False)
        self.dwc = InceptionDWG(in_dim, in_dim, 3, 'same')
        self.act = nn.GELU()

    def forward(self, input):
        out = self.conv1(input)
        out = self.dwc(out)
        return out

# Multi-scale Residual Block
class MSRB(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.dim = in_dim // 4
        self.conv1 = nn.Conv2d(in_dim, in_dim, 1,1,0,bias=False)
        self.dwc = InceptionDWG(in_dim, in_dim, 3, 'same')
        self.act = nn.GELU()

    def forward(self, input):
        out = self.conv1(input)
        out = self.dwc(out) + input
        out = self.act(out)
        return out

######  Inception Depth-Wise Convolution Group  ######
class InceptionDWG(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.dim = in_dim // 4
        self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding,
                                 groups=in_dim // 4,bias=False)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size * 2 - 1, kernel_size), padding=padding,
                                 groups=in_dim // 4,bias=False)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size * 2 - 1), padding=padding,
                                 groups=in_dim // 4,bias=False)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, kernel_size=kernel_size, padding=padding, groups=in_dim // 4,bias=False),
            nn.Conv2d(in_dim // 4, out_dim // 4, kernel_size=kernel_size, padding=padding, groups=in_dim // 4,bias=False)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)

    def forward(self, input):
        out = input
        out = torch.cat([self.conv2_1(out[:, :self.dim, :, :]), self.conv2_2(out[:, self.dim:self.dim * 2, :, :]),
                         self.conv2_3(out[:, self.dim * 2:self.dim * 3, :, :]),
                         self.conv2_4(out[:, self.dim * 3:, :, :])], dim=1)
        return out
    

# Spatial Channel Attention Block
class SCAB(nn.Module):
    def __init__(self, in_channels):
        super(SCAB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = MSRB(in_channels)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = MSRB(in_channels)

        self.c3 = nn.Conv2d(self.remaining_channels, self.dc, 3, 1, 1, groups=self.dc)
        self.act = nn.GELU()

        self.c4 = nn.Conv2d(self.dc * 3, in_channels, 1)

        self.local_handle = MSRB(in_channels)
        self.spatial_attn = VCSA(in_channels)
        self.channel_attn = GCAB(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)

        r_c3 = self.act(self.c3(r_c2))

        out = torch.cat([distilled_c1, distilled_c2, r_c3], dim=1)
        out = self.c4(out)

        out = self.local_handle(out)
        out = self.spatial_attn(out)
        out = self.channel_attn(out)

        return out + input

# Variance-Guided Spatial-Channel Context Interaction Network
class VSCINet(nn.Module):
    def __init__(self, dim=20, upscale=4):
        super().__init__()

        # MeanShift for Image Input
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=False)

        self.B1 = SCAB(dim)
        self.B2 = SCAB(dim)
        self.B3 = SCAB(dim)
        self.B4 = SCAB(dim)
        self.B5 = SCAB(dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 5, dim, 1, 1, 0, bias=False),
            nn.GELU(),
            MSConv(dim)
        )


        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscale ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = x - self.mean

        x0 = self.to_feat(x)

        x1 = self.B1(x0)
        x2 = self.B2(x1)
        x3 = self.B3(x2)
        x4 = self.B4(x3)
        x5 = self.B5(x4)

        out = torch.cat([x1, x2, x3, x4, x5 ], dim=1)
        out = self.fuse(out) + x0
        out = self.to_img(out)

        out = out + self.mean
        return out


if __name__ == '__main__':
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    x = torch.randn(1, 3, 256, 256).cuda()

    model = VSCINet(dim=20, upscale=4).cuda()

    print(model)
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    output = model(x)
    print(output.shape)