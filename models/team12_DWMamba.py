import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange, repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)
    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)
    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

def window_partition(x, window_size):
    b, h, w, c = x.shape
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
        
    x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], c)
    return windows

def window_reverse(windows, window_size, h, w):
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
        
    b = int(windows.shape[0] / (h * w / window_size[0] / window_size[1]))
    x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x

class DWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False  

    def forward(self, x):
        """
        x: (B, C, H, W)
        Returns:
            LL: (B, C, H/2, W/2)
            HF: (B, 3C, H/2, W/2) - Stacked LH, HL, HH
        """
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        
        x_HF = torch.cat([x_HL, x_LH, x_HH], dim=1)
        return x_LL, x_HF

class IDWT(nn.Module):
    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, LL, HF):
        """
        LL: (B, C, H/2, W/2)
        HF: (B, 3C, H/2, W/2)
        Returns:
            x: (B, C, H, W)
        """
        b, c, h, w = LL.shape

        HL, LH, HH = torch.chunk(HF, 3, dim=1)

        x1 = (LL - HL - LH + HH) 
        x2 = (LL - HL + LH - HH) 
        x3 = (LL + HL - LH - HH) 
        x4 = (LL + HL + LH + HH) 
        
        y_top = torch.stack((x1, x3), dim=-1).reshape(b, c, h, w*2)
        y_bot = torch.stack((x2, x4), dim=-1).reshape(b, c, h, w*2)
        y = torch.stack((y_top, y_bot), dim=-2).reshape(b, c, h*2, w*2)
        
        return y/2

class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V 



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class HFBranch(nn.Module):
    """
    Detail-Preserving CNN for High Frequency Components.
    Input: HF features (B, 3C, H/2, W/2)
    Output: Processed HF features (B, 3C, H/2, W/2)
    """
    def __init__(self, in_channels, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, 1, 1, 0) 
        self.se = SEBlock(in_channels,reduction=4)
        
        #self.norm = nn.GroupNorm(4, num_channels=in_channels)
        self.res_scale = nn.Parameter(torch.ones(1, in_channels, 1, 1) * res_scale, requires_grad=True)
    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.se(x)


        return x * self.res_scale + shortcut


class CFI(nn.Module):
    def __init__(self, ll_dim, hf_dim):
        super().__init__()
        self.struct_conv = nn.Conv2d(ll_dim, hf_dim, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()
        self.fusion = nn.Conv2d(hf_dim * 2, hf_dim, 3, 1, 1)
        
    def forward(self, ll_feat, hf_feat):
        ll_proj = self.struct_conv(ll_feat.contiguous())  # (B, hf_dim, H/2, W/2)
        mask = self.sigmoid(ll_proj)
        hf_gated = hf_feat * mask
        hf_out = hf_gated + self.fusion(torch.cat([hf_feat, ll_proj], dim=1))
        
        return hf_out

class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)  

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2
            
class GatedMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        H, W = x_size
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class Selective_Scan(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2., dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) + prompt
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(xs, dts, As, Bs, Cs, Ds, z=None, delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False).view(B, K, -1, L)
        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)
        y = y.permute(0, 2, 1).contiguous()
        return y

class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.in_proj = nn.Sequential(nn.Conv2d(self.dim, hidden, 1, 1, 0))


        self.CPE = nn.Sequential(nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden))
        self.hf_dim=dim//4
        self.hf_dim=max(self.hf_dim & (~3), 12) # 保持hf_dim是4的倍数，且不小于12
        self.cpe_hf_proj = nn.Conv2d(self.hf_dim, hidden, 1, 1, 0)

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)
        self.route = nn.Sequential(
            nn.Linear(self.dim*2, (self.dim * 2 )// 3),
            nn.GELU(),
            nn.Linear((self.dim*2) // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

        self.hf_align=nn.Sequential(
            nn.Conv2d(self.hf_dim, self.hf_dim, 3, 1, 1, groups=self.hf_dim),
            nn.SiLU(),
            nn.Conv2d(self.hf_dim, self.dim, 1, 1, 0)           
        )

        self.hf_gate_route = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        self.hf_gate_cpe = nn.Parameter(torch.zeros(1,hidden,1,1), requires_grad=True)

    def forward(self, x, x_size, token,x_hf=None):
        B, n, C = x.shape
        H, W = x_size
        full_embedding = self.embeddingB.weight @ token.weight
        hf=self.hf_align(x_hf).flatten(2).permute(0,2,1)
        route_input=torch.cat([ x,hf * self.hf_gate_route],dim=-1)
        pred_route = self.route(route_input)
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)
        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        gate_input = self.CPE(x) + self.cpe_hf_proj(x_hf) * self.hf_gate_cpe

        x = x * torch.sigmoid(gate_input)
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)
        semantic_x = semantic_neighbor(x, x_sort_indices)
        prompt = semantic_neighbor(prompt, x_sort_indices)
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse)
        return x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, groups=hidden_features),
            nn.GELU()
        )
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.proj = nn.Linear(dim, dim)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            attn_area, attn_area, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

class AttentiveLayer(nn.Module):
    """
    Standard MambaIRv2 Layer. Used for LL branch.
    Input resolution will be H/2 * W/2.
    """
    def __init__(self, dim, d_state, input_resolution, num_heads, window_size, shift_size, inner_rank, num_tokens, convffn_kernel_size, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.scale1 = nn.Parameter(1e-4 * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(1e-4 * torch.ones(dim), requires_grad=True)
        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.win_mhsa = WindowAttention(self.dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, qkv_bias=qkv_bias)
        self.assm = ASSM(self.dim, d_state, input_resolution=input_resolution, num_tokens=num_tokens, inner_rank=inner_rank, mlp_ratio=mlp_ratio)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn1 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )
        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, )
        self.embeddingA = nn.Embedding(inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / inner_rank, 1 / inner_rank)

    def forward(self, x, x_size, params, x_hf=None):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c
        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)
        qkv = qkv.reshape(b, h, w, c3)
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None
        x_windows = window_partition(shifted_qkv, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
        attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x.view(b, n, c) + shortcut
        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        shortcut = x
        x_aca = self.assm(self.norm3(x), x_size, self.embeddingA, x_hf=x_hf) + x
        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x
        return x

class WaveletASSB(nn.Module):
    """
    W-ASSM Block: The core innovation of WaveMambaIR.
    Integrates DWT, Dual-Branch (Mamba LL + CNN HF), CFI, and IDWT.
    """
    def __init__(self, dim, d_state, input_resolution, num_heads, window_size, inner_rank, num_tokens, convffn_kernel_size, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        
        # 1. DWT & IDWT
        self.dwt = DWT()
        self.idwt = IDWT()
        
        # 2. LL Branch: AttentiveLayer (Mamba + Window Attn)
        ll_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        self.ll_branch = AttentiveLayer(
            dim=dim, 
            d_state=d_state,
            input_resolution=ll_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer
        )
        

        self.skff=SKFF(in_channels=dim, height=3, reduction=8)

        
        self.hf_dim = dim//4
        self.hf_dim=max(self.hf_dim & (~3), 12)
        self.hf_compress = nn.Conv2d(dim, self.hf_dim, 1, 1, 0)

        # 3. HF Branch: DP-CNN
        self.hf_branch = HFBranch(in_channels=self.hf_dim, res_scale=0.1)
        
        # 4. Interaction: CFI
        self.cfi=CFI(ll_dim=dim, hf_dim=self.hf_dim)
        
        self.expand_conv = nn.Conv2d(self.hf_dim, 3*dim, 1, 1, 0)
        nn.init.constant_(self.expand_conv.bias, 0)

    def forward(self, x, x_size, params_ll,return_freq=False):
        """
        x: (B, N, C) - Flattened input
        x_size: (H, W)
        params_ll: Dict containing masks/rpi for the Low-Resolution (H/2, W/2) branch
        """
        H, W = x_size
        B, N, C = x.shape
        x_img = x.transpose(1, 2).view(B, C, H, W)
        
        x_ll, x_hf = self.dwt(x_img) 
        
        hf_subbands=torch.chunk(x_hf, 3, dim=1) 
        x_hf_fused=self.skff(hf_subbands) 

        x_hf_fused = self.hf_compress(x_hf_fused.contiguous())
        x_hf_out = self.hf_branch(x_hf_fused) 

        x_ll_flat = x_ll.flatten(2).transpose(1, 2) 
        ll_size = (H // 2, W // 2)
        x_ll_out_flat = self.ll_branch(x_ll_flat, ll_size, params_ll,x_hf_out)
        x_ll_out = x_ll_out_flat.transpose(1, 2).contiguous().view(B, C, H//2, W//2)

        x_hf_refined = self.cfi(x_ll_out, x_hf_out)
        
        x_final=self.expand_conv(x_hf_refined) 
        x_rec = self.idwt(x_ll_out, x_final) 
        
        x_out = x_rec # + x_img
        
        x_out_flat = x_out.flatten(2).transpose(1, 2)

        if return_freq:
            return x_out_flat, x_ll_out, x_final
        return x_out_flat

class WaveMambaIRBlockGroup(nn.Module):
    """
    Replaces BasicBlock. Stacks WaveletASSB.
    """
    def __init__(self, 
                 dim, 
                 d_state, 
                 input_resolution, 
                 depth, 
                 num_heads, 
                 window_size, 
                 inner_rank, 
                 num_tokens, 
                 convffn_kernel_size, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 img_size=64,
                 patch_size=1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=0, 
            embed_dim=dim, 
            norm_layer=None)
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=0, 
            embed_dim=dim, 
            norm_layer=None)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                WaveletASSB(
                    dim=dim,
                    d_state=d_state,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    inner_rank=inner_rank,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    shift_size=0 if (i % 2 == 0) else window_size // 2
                )
            )

    def forward(self, x, x_size, params_ll,return_freq=False):
        shortcut=x
        x_ll,x_hf=None,None
        for i,layer in enumerate(self.layers):
            if return_freq and i==len(self.layers)-1:
                x ,x_ll,x_hf= layer(x, x_size, params_ll,return_freq=True)
            else:
                x = layer(x, x_size, params_ll,return_freq=False)
        x = self.patch_unembed(x, x_size) # -> (B, C, H, W)
        x = self.conv(x)                  # -> Conv
        x = self.patch_embed(x)
        out=x+shortcut
        if return_freq:
            return out, x_ll, x_hf
        return out

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])
        return x

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported.')
        super(Upsample, self).__init__(*m)

class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

# @ARCH_REGISTRY.register()
class DWMamba(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=48,
                 d_state=8,
                 depths=(6, 6, 6, 6,),
                 num_heads=(4, 4, 4, 4,),
                 window_size=16,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction (WaveMamba Backbone)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=embed_dim, 
            embed_dim=embed_dim, 
            norm_layer=norm_layer if self.patch_norm else None
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.register_buffer('relative_position_index_SA', self.calculate_rpi_sa())

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = WaveMambaIRBlockGroup(
                dim=embed_dim,
                d_state=d_state,
                input_resolution=tuple(patches_resolution),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # 3. Reconstruction
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask!= 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, return_freq=False):
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size * 2 # Multiply by 2 because DWT downsizes by 2
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        ll_size = (h // 2, w // 2)
        attn_mask_ll = self.calculate_mask(ll_size).to(x.device)
        params_ll = {'attn_mask': attn_mask_ll, 'rpi_sa': self.relative_position_index_SA}

        # 1. Shallow
        x = self.conv_first(x)
        
        # 2. Body (WaveMamba)
        x_res = self.patch_embed(x)
        if self.ape:
            x_res = x_res + self.absolute_pos_embed

        x_ll, x_hf = None, None
        for i, layer in enumerate(self.layers):
            if return_freq and i==len(self.layers)-1:
                x_res, x_ll, x_hf = layer(x_res, (h, w), params_ll,return_freq=True)
            else:
                x_res = layer(x_res, (h, w), params_ll,return_freq=False)
            
        x_res = self.norm(x_res)
        x_res = self.patch_unembed(x_res, (h, w))
        
        x = x + self.conv_after_body(x_res)

        # 3. Reconstruction
        if self.upsampler == 'pixelshuffle':
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x = x + self.conv_last(x)

        x = x / self.img_range + self.mean
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        if return_freq:
            return x, x_ll, x_hf
        return x

if __name__ == '__main__':
    upscale = 4
    model = DWMamba(
        upscale=2,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[4, 4, 4, 4], 
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        img_range=1.,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect').cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    
    _input = torch.randn(1, 3, 64, 64).cuda()
    output = model(_input).cuda()
    print("Output Shape:", output.shape)
    
    print("\n--- DWT Check ---")
    dwt = DWT().cuda()
    ll, hf = dwt(_input)
    print(f"LL shape: {ll.shape} (Expected: )")
    print(f"HF shape: {hf.shape} (Expected: )")

    