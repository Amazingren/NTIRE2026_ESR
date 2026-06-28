from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from mamba_ssm.modules.mamba_simple import Mamba

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

import torch
import torch.nn as nn

try:
    # mamba-ssm (常见导入路径之一)
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    Mamba = None


class MambaGate2D(nn.Module):
    """
    输入:  [B, C, H, W]
    输出:  sim_att [B, C, H, W]  (范围约在 [-0.5, 0.5])
    """
    def __init__(
        self,
        channels: int,
        d_model: int = 64,      # Mamba内部宽度（越小越轻）
        d_state: int = 16,      # 状态维度（越小越轻）
        expand: int = 2,
        bidir: bool = True,     # 双向扫描更稳（会稍微慢一点）
    ):
        super().__init__()
        assert Mamba is not None, "请先安装 mamba-ssm，或替换为你自己的Mamba实现。"

        self.channels = channels
        self.d_model = d_model
        self.bidir = bidir

        # 轻量：先 1x1 降维到 d_model，再做 Mamba
        self.in_proj = nn.Conv2d(channels, d_model, kernel_size=1, bias=True)

        # LayerNorm 用在 token 维度上（[B, L, d_model]）
        self.norm = nn.LayerNorm(d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
        )

        self.out_proj = nn.Conv2d(d_model, channels, kernel_size=1, bias=True)

        # 让门控初始接近 0：sigmoid(0)-0.5=0（更贴近你原先“无参门控”的初态）
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        y = self.in_proj(x)  # [B, d_model, H, W]

        # -> tokens: [B, L, d_model], L = H*W
        tokens = y.permute(0, 2, 3, 1).contiguous().view(b, h * w, self.d_model)
        tokens = self.norm(tokens)

        if self.bidir:
            # 双向：正向 + 反向（反向需要翻转序列维）
            out_f = self.mamba(tokens)
            out_b = self.mamba(torch.flip(tokens, dims=[1]))
            out_b = torch.flip(out_b, dims=[1])
            out = 0.5 * (out_f + out_b)
        else:
            out = self.mamba(tokens)

        # tokens -> [B, d_model, H, W]
        out2d = out.view(b, h, w, self.d_model).permute(0, 3, 1, 2).contiguous()
        gate_logits = self.out_proj(out2d)  # [B, C, H, W]

        sim_att = torch.sigmoid(gate_logits) - 0.5
        return sim_att

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

        self._fused = False  # ✅ 新增
        self.update_params()  # 初始化先 fuse 一次（在当前 device 上）
        self._fused = True  # ✅ 新增
        # self.update_params()

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
            self._fused = False  # ✅ 训练时让它失效（因为权重在变）
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            if not self._fused:  # ✅ eval 时只 fuse 一次
                self.update_params()
                self._fused = True
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

class SPAB_mamba(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False,
                 mamba_d_model: int = 32,
                 mamba_d_state: int = 16,
                 mamba_expand: int = 2,
                 mamba_bidir: bool = True,
                 ):
        super(SPAB_mamba, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=1, s=1, bias=bias)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=1, s=1, bias=bias)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=1, s=1, bias=bias)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)
        self.mamba_gate = MambaGate2D(
            channels=out_channels,
            d_model=min(64, out_channels),
            d_state=mamba_d_state,
            expand=mamba_expand,
            bidir=mamba_bidir,
        )

    def forward(self, x):
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = self.mamba_gate(out3)

        out = out3 + x
        out = out + out * sim_att  # ✅ 不会初始全 0

        return out, out1, sim_att


@ARCH_REGISTRY.register()
class SPANMamba_single_1_4T05(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=48,
                 upscale=4,
                 bias=True,
                 img_range=1.0,
                 rgb_mean=(0.4488, 0.4371, 0.4040),
                 mamba_bidir=False
                 ):
        super(SPANMamba_single_1_4T05, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = Conv3XC(in_channels, feature_channels, gain1=1, s=1)
        self.block_1 = SPAB_mamba(feature_channels, bias=bias, mamba_d_model=32, mamba_d_state=16, mamba_expand=2,
                                  mamba_bidir=mamba_bidir)
        self.block_2 = SPAB_mamba(feature_channels, bias=bias, mamba_d_model=32, mamba_d_state=16, mamba_expand=2,
                                  mamba_bidir=mamba_bidir)
        self.block_3 = SPAB_mamba(feature_channels, bias=bias, mamba_d_model=32, mamba_d_state=16, mamba_expand=2,
                                  mamba_bidir=mamba_bidir)
        self.block_4 = SPAB_mamba(feature_channels, bias=bias, mamba_d_model=32, mamba_d_state=16, mamba_expand=2,
                                  mamba_bidir=mamba_bidir)
        self.block_5 = SPAB_mamba(feature_channels, bias=bias, mamba_d_model=32, mamba_d_state=16, mamba_expand=2,
                                  mamba_bidir=mamba_bidir)
        self.block_6 = SPAB_mamba(feature_channels, bias=bias, mamba_d_model=32, mamba_d_state=16, mamba_expand=2,
                                  mamba_bidir=mamba_bidir)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, feature_channels, gain1=1, s=1)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, att1 = self.block_1(out_feature)
        out_b2, _, att2 = self.block_2(out_b1)
        out_b3, _, att3 = self.block_3(out_b2)

        out_b4, _, att4 = self.block_4(out_b3)
        out_b5, _, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        # 之前是out_b5_2，改成与论文一致的相加——out_b5
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5], 1))
        output = self.upsampler(out)

        return output

if __name__ == "__main__":
    import time
    import torch

    # --------- build model ----------
    model = SPANMamba(3, 3, upscale=4, feature_channels=48).cuda()
    model.eval()

    H, W = 256, 256

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params / 1e6:.3f} M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
    print(f"Param memory (fp32): {total_params * 4 / 1024 / 1024:.2f} MB")
    # --------- print output info with batch=1 ----------
    x1 = torch.rand(1, 3, H, W, device="cuda")
    with torch.no_grad():
        out = model(x1)

    if isinstance(out, (tuple, list)):
        print("Model outputs a tuple/list, length =", len(out))
        for i, o in enumerate(out):
            if torch.is_tensor(o):
                print(f"  out[{i}] shape={tuple(o.shape)} dtype={o.dtype} device={o.device}")
            else:
                print(f"  out[{i}] type={type(o)}")
    else:
        print("Output shape:", tuple(out.shape))
        print("dtype:", out.dtype, "device:", out.device)


    # --------- benchmark function ----------
    def bench(batch_size: int, iters: int = 50, warmup: int = 10):
        x = torch.rand(batch_size, 3, H, W, device="cuda")
        with torch.no_grad():
            # warmup
            for _ in range(warmup):
                _ = model(x)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(iters):
                _ = model(x)
            torch.cuda.synchronize()
            t1 = time.time()

        total_ms = (t1 - t0) * 1000
        avg_ms_per_iter = total_ms / iters
        avg_ms_per_img = avg_ms_per_iter / batch_size
        return avg_ms_per_iter, avg_ms_per_img


    # --------- run benchmarks ----------
    print("\n=== Runtime Benchmark ===")
    for bs in [8, 16, 32]:
        try:
            avg_iter_ms, avg_img_ms = bench(batch_size=bs, iters=50, warmup=10)
            print(f"[BS={bs:2d}] avg: {avg_iter_ms:.3f} ms/iter | {avg_img_ms:.3f} ms/image")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[BS={bs:2d}] CUDA OOM -> skip")
                torch.cuda.empty_cache()
            else:
                raise



