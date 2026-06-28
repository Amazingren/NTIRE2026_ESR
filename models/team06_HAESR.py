import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
from torch import einsum
from einops import rearrange

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3,
                       bias=False):
    """
    Upsample features according to `upscale_factor`.
    """
    padding = kernel_size // 2
    conv = nn.Conv2d(in_channels,
                     out_channels * (upscale_factor ** 2),
                     kernel_size,
                     padding=1,
                     bias=bias)  
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


################################################
class Gated_Conv_FeedForward(nn.Module):
    def __init__(self, dim, mult = 1, bias=False, dropout = 0.):
        super().__init__()

        hidden_features = int(dim*mult)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class CA(nn.Module):
    def __init__(self, channels):
        super(CA, self).__init__()
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.AdaptiveAvgPool(x))
        out = out * x
        return out


class PLKB(nn.Module):
    def __init__(self, channels, large_kernel, split_group):
        super(PLKB, self).__init__()
        self.channels = channels
        self.split_group = split_group
        self.split_channels = int(channels // split_group)
        self.CA = CA(channels)
        self.DWConv_Kx1 = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(large_kernel, 1), stride=1,
                                    padding=(large_kernel // 2, 0), groups=self.split_channels)
        self.DWConv_1xK = nn.Conv2d(self.split_channels, self.split_channels, kernel_size=(1, large_kernel), stride=1,
                                    padding=(0, large_kernel // 2), groups=self.split_channels)
        self.conv1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.reshape(B, self.split_channels, self.split_group, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, C, H, W)

        x1, x2 = torch.split(x, (self.split_channels, self.channels - self.split_channels), dim=1)
        x1 = self.CA(x1)

        x1 = self.DWConv_Kx1(self.DWConv_1xK(x1))
        out = torch.cat((x1, x2), dim=1)
        out = self.act(self.conv1(out))
        return out


class HFAB(nn.Module):
    def __init__(self, channels, large_kernel, split_group):
        super(HFAB, self).__init__()
        self.PLKB = PLKB(channels, large_kernel, split_group)
        self.DWConv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels)
        self.conv1 = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.DWConv3(x)
        x2 = self.PLKB(x)
        out = self.act(self.conv1(torch.cat((x1, x2), dim=1)))
        return out


class ExtendedWindowUnfold(nn.Module):
    def __init__(self, window_size=8, extend_size=4, sample_pattern='star'):
        super().__init__()
        self.w = window_size
        self.ext = extend_size
        self.pattern = sample_pattern
        self.ext_size = window_size + 2 * extend_size
        
        self.unfold = nn.Unfold(
            kernel_size=(self.ext_size, self.ext_size),
            stride=window_size,
            padding=extend_size
        )
        
        self.register_buffer('sample_indices', self._create_sample_indices())
        self.register_buffer('center_indices', self._create_center_indices())
        
    def _create_center_indices(self):
        w = self.w
        ext = self.ext
        ew = self.ext_size
        indices = []
        for i in range(w):
            for j in range(w):
                idx = (ext + i) * ew + (ext + j)
                indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)
        
    def _create_sample_indices(self):
        w = self.w
        ext = self.ext
        ew = self.ext_size
        
        indices = []
        
        for i in range(w):
            for j in range(w):
                idx = (ext + i) * ew + (ext + j)
                indices.append(idx)
        
        if self.pattern == 'star':
            col = ext + w // 2
            for step in range(ext):
                row = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext + w + step
                indices.append(row * ew + col)
            row = ext + w // 2
            for step in range(ext):
                col = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                col = ext + w + step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext - 1 - step
                col = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext - 1 - step
                col = ext + w + step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext + w + step
                col = ext - 1 - step
                indices.append(row * ew + col)
            for step in range(ext):
                row = ext + w + step
                col = ext + w + step
                indices.append(row * ew + col)
                
        elif self.pattern == 'cross':
            row = ext - 1
            for step in range(ext):
                r = row - step
                for col_offset in [-1, 0]:
                    c = ext + w // 2 + col_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
            row = ext + w
            for step in range(ext):
                r = row + step
                for col_offset in [-1, 0]:
                    c = ext + w // 2 + col_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
            col = ext - 1
            for step in range(ext):
                c = col - step
                for row_offset in [-1, 0]:
                    r = ext + w // 2 + row_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
            
            col = ext + w
            for step in range(ext):
                c = col + step
                for row_offset in [-1, 0]:
                    r = ext + w // 2 + row_offset
                    if 0 <= r < ew and 0 <= c < ew:
                        indices.append(r * ew + c)
                
        elif self.pattern == 'corner':
            corner_size = 3
            
            corner_starts = [
                (0, 0),
                (0, ew - corner_size),
                (ew - corner_size, 0),
                (ew - corner_size, ew - corner_size),
            ]
            for start_r, start_c in corner_starts:
                for i in range(corner_size):
                    for j in range(corner_size):
                        r = start_r + i
                        c = start_c + j
                        if not (ext <= r < ext + w and ext <= c < ext + w):
                            indices.append(r * ew + c)
        
        return torch.tensor(sorted(set(indices)), dtype=torch.long)
    
    def forward(self, x):
        B, C, H, W = x.shape
        w = self.w
        ew = self.ext_size
        
        nH, nW = H // w, W // w
        NW = nH * nW
        
        unfolded = self.unfold(x)
        
        unfolded = unfolded.view(B, C, ew * ew, NW)
        unfolded = unfolded.permute(0, 3, 2, 1)
        
        center = unfolded[:, :, self.center_indices, :]
        extended = unfolded[:, :, self.sample_indices, :]
        
        return center, extended, (nH, nW)


class SelfAttentionA_Extended(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        extend_size=2,
        sample_pattern='star',
        with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
        
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.w = window_size
        self.with_pe = with_pe
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        
        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        self.ext_unfold = ExtendedWindowUnfold(window_size, extend_size, sample_pattern)
        
        if self.with_pe:
            num_samples = len(self.ext_unfold.sample_indices)
            num_q = window_size * window_size
            self.rel_pos_bias = nn.Parameter(torch.zeros(self.heads, num_q, num_samples))
            nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        center, extended, (nH, nW) = self.ext_unfold(x)
        
        NW = nH * nW
        q = self.to_q(center)
        kv = self.to_kv(extended)
        k, v = kv.chunk(2, dim=-1)
        
        q = rearrange(q, 'b n s (h d) -> (b n) h s d', h=h)
        k = rearrange(k, 'b n s (h d) -> (b n) h s d', h=h)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        
        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        if self.with_pe:
            sim = sim + self.rel_pos_bias.unsqueeze(0)
        
        attn = self.attend(sim)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=NW)
        out = self.to_out(out)
        out = rearrange(out, 'b (nh nw) (w1 w2) c -> b c (nh w1) (nw w2)', 
                       nh=nH, nw=nW, w1=w, w2=w)
        return out, attn


class SelfAttentionB_Extended(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        extend_size=2,
        sample_pattern='star',
    ):
        super().__init__()
        assert (dim % dim_head) == 0
        
        self.heads = dim // dim_head
        self.w = window_size
        
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )
        
        self.ext_unfold = ExtendedWindowUnfold(window_size, extend_size, sample_pattern)
    
    def forward(self, x, attn):
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        _, extended, (nH, nW) = self.ext_unfold(x)
        NW = nH * nW
        
        v = self.to_v(extended)
        v = rearrange(v, 'b n s (h d) -> (b n) h s d', h=h)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, '(b n) h s d -> b n s (h d)', b=B, n=NW)
        
        out = self.to_out(out)
        out = rearrange(out, 'b (nh nw) (w1 w2) c -> b c (nh w1) (nw w2)',
                       nh=nH, nw=nW, w1=w, w2=w)
        
        return out


class GridCenterUnfold(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_base = window_size * window_size
        self.num_extra = (window_size - 1) * (window_size - 1)
        self.num_kv = self.num_base + self.num_extra
        
    def forward(self, x):
        B, C, H, W = x.shape
        w = self.w
        nH, nW = H // w, W // w
        
        x_grid = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        q_tokens = rearrange(x_grid, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        offset_h, offset_w = nH // 2, nW // 2
        
        x_padded = F.pad(x, (0, offset_w, 0, offset_h), mode='reflect')
        
        x_offset = x_padded[:, :, offset_h:offset_h+H, offset_w:offset_w+W]
        
        x_center_grid = rearrange(x_offset, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        
        centers = x_center_grid[:, :, :, :w-1, :w-1, :]
        centers = rearrange(centers, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        kv_tokens = torch.cat([q_tokens, centers], dim=1)
        
        return q_tokens, kv_tokens, (nH, nW)


class GridRowMidUnfold(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_base = window_size * window_size
        self.num_extra = window_size * (window_size - 1)
        self.num_kv = self.num_base + self.num_extra
        
    def forward(self, x):
        B, C, H, W = x.shape
        w = self.w
        nH, nW = H // w, W // w
        
        x_grid = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        q_tokens = rearrange(x_grid, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        offset_w = nW // 2
        
        x_padded = F.pad(x, (0, offset_w, 0, 0), mode='reflect')
        x_offset = x_padded[:, :, :, offset_w:offset_w+W]
        
        x_row_grid = rearrange(x_offset, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        
        row_mids = x_row_grid[:, :, :, :, :w-1, :]
        row_mids = rearrange(row_mids, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        kv_tokens = torch.cat([q_tokens, row_mids], dim=1)
        
        return q_tokens, kv_tokens, (nH, nW)


class GridColMidUnfold(nn.Module):
    def __init__(self, window_size=8):
        super().__init__()
        self.w = window_size
        self.num_base = window_size * window_size
        self.num_extra = (window_size - 1) * window_size
        self.num_kv = self.num_base + self.num_extra
        
    def forward(self, x):
        B, C, H, W = x.shape
        w = self.w
        nH, nW = H // w, W // w

        x_grid = rearrange(x, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)
        q_tokens = rearrange(x_grid, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        

        offset_h = nH // 2

        x_padded = F.pad(x, (0, 0, 0, offset_h), mode='reflect')
        x_offset = x_padded[:, :, offset_h:offset_h+H, :]
        
        x_col_grid = rearrange(x_offset, 'b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w)

        col_mids = x_col_grid[:, :, :, :w-1, :, :]
        col_mids = rearrange(col_mids, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        
        kv_tokens = torch.cat([q_tokens, col_mids], dim=1)
        
        return q_tokens, kv_tokens, (nH, nW)


class SelfAttentionA_GridSample(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        sample_mode='center',
        with_pe=True,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.w = window_size
        self.with_pe = with_pe
        self.sample_mode = sample_mode

        if sample_mode == 'center':
            self.grid_unfold = GridCenterUnfold(window_size)
        elif sample_mode == 'row_mid':
            self.grid_unfold = GridRowMidUnfold(window_size)
        elif sample_mode == 'col_mid':
            self.grid_unfold = GridColMidUnfold(window_size)
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")
        
        self.num_base = self.grid_unfold.num_base
        self.num_extra = self.grid_unfold.num_extra
        self.num_kv = self.grid_unfold.num_kv

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)

        self.attend = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

        if self.with_pe:
            self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
            pos = torch.arange(window_size)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing='ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
            rel_pos += window_size - 1
            rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
            self.register_buffer('rel_pos_indices', rel_pos_indices, persistent=False)
            
            self.extra_pos_bias = nn.Parameter(torch.zeros(self.heads, self.num_base, self.num_extra))
            nn.init.trunc_normal_(self.extra_pos_bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        nH, nW = H // w, W // w
        q_tokens, kv_tokens, _ = self.grid_unfold(x)
        
        q = self.to_q(q_tokens)
        kv = self.to_kv(kv_tokens)
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        
        if self.with_pe:
            bias_base = self.rel_pos_bias(self.rel_pos_indices)
            bias_base = rearrange(bias_base, 'i j h -> h i j')
            bias_extra = self.extra_pos_bias
            bias = torch.cat([bias_base, bias_extra], dim=-1)
            sim = sim + bias.unsqueeze(0)

        # Attention
        attn = self.attend(sim)

        # Aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=w, w2=w)
        out = self.to_out(out)
        out = rearrange(out, '(b x y) w1 w2 d -> b d (w1 x) (w2 y)', x=nH, y=nW)
        return out, attn


class SelfAttentionB_GridSample(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=32,
        dropout=0.,
        window_size=8,
        sample_mode='center',
    ):
        super().__init__()
        assert (dim % dim_head) == 0

        self.heads = dim // dim_head
        self.w = window_size
    
        if sample_mode == 'center':
            self.grid_unfold = GridCenterUnfold(window_size)
        elif sample_mode == 'row_mid':
            self.grid_unfold = GridRowMidUnfold(window_size)
        elif sample_mode == 'col_mid':
            self.grid_unfold = GridColMidUnfold(window_size)
        else:
            raise ValueError(f"Unknown sample_mode: {sample_mode}")

        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn):
        B, C, H, W = x.shape
        h = self.heads
        w = self.w
        
        nH, nW = H // w, W // w
        _, kv_tokens, _ = self.grid_unfold(x)

        v = self.to_v(kv_tokens)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1=w, w2=w)

        out = self.to_out(out)
        out = rearrange(out, '(b x y) w1 w2 d -> b d (w1 x) (w2 y)', x=nH, y=nW)
        return out


class Channel_Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)


        out = self.project_out(out)

        return out
        

class Channel_Attention_grid(nn.Module):
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)


        out = self.project_out(out)

        return out


class SLA_A_Ext(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True, 
                 window_size=8, extend_size=2, sample_pattern='star',
                 with_pe=False, dropout=0.0):
        super(SLA_A_Ext, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = SelfAttentionA_Extended(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, extend_size=extend_size,
            sample_pattern=sample_pattern, with_pe=with_pe
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x):
        attn_out, a = self.attn(self.cnorm(x))
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x, a


class SGA_A_Sample(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, sample_mode='center', with_pe=False, dropout=0.0):
        super(SGA_A_Sample, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = SelfAttentionA_GridSample(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, sample_mode=sample_mode, with_pe=with_pe
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x):
        attn_out, a = self.attn(self.cnorm(x))
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x, a


class SLA_B_Ext(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, extend_size=2, sample_pattern='star',
                 with_pe=False, dropout=0.0):
        super(SLA_B_Ext, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = SelfAttentionB_Extended(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, extend_size=extend_size,
            sample_pattern=sample_pattern
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x, a):
        attn_out = self.attn(self.cnorm(x), a)
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x


class SGA_B_Sample(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, sample_mode='center', with_pe=False, dropout=0.0):
        super(SGA_B_Sample, self).__init__()
        self.w = window_size
        
        self.cnorm = LayerNorm2d(channel_num)
        self.attn = SelfAttentionB_GridSample(
            dim=channel_num, dim_head=channel_num, dropout=dropout,
            window_size=window_size, sample_mode=sample_mode
        )
        self.cnorm2 = LayerNorm2d(channel_num)
        self.gfn = Gated_Conv_FeedForward(dim=channel_num, dropout=dropout)

    def forward(self, x, a):
        attn_out = self.attn(self.cnorm(x), a)
        x = x + attn_out
        x = self.gfn(self.cnorm2(x)) + x
        return x


class TrBlockC(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True, window_size=8, with_pe=False, dropout=0.0):
        super(TrBlockC, self).__init__()

        self.w= 8
        
        ####
                
        self.canorm1 = LayerNorm2d(channel_num) #
        self.canorm2 = LayerNorm2d(channel_num) #
        
        self.cttn1 = Channel_Attention(dim = channel_num, heads=1, dropout = dropout, window_size = window_size) #
        self.cttn2 = Channel_Attention_grid(dim = channel_num, heads=1, dropout = dropout, window_size = window_size) #
        
        ####
        
        self.cnorm1 = LayerNorm2d(channel_num)
        self.cnorm2 = LayerNorm2d(channel_num) #
       
        self.gfn1 = Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)
        self.gfn2 = Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)#   

    def forward(self, x):

        #
        x = self.cttn1(self.canorm1(x))+x
        x = self.gfn1(self.cnorm1(x))+x
        #
        x = self.cttn2(self.canorm2(x))+x
        x = self.gfn2(self.cnorm2(x))+x
        #

        return x


class IDSA_Center_Block1(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True, 
                 window_size=8, extend_size=4, sample_pattern='star',
                 sga_modes=None, with_pe=False, dropout=0.0):
        super(IDSA_Center_Block1, self).__init__()

        self.w = 8
        self.ch_split = channel_num // 4
        
        if sga_modes is None:
            sga_modes = ['center', 'row_mid', 'col_mid']
        
        # Local Blocks
        self.LB1 = HFAB(channel_num, large_kernel=31, split_group=4)
        self.LB2 = HFAB(channel_num//4*3, large_kernel=31, split_group=4)
        self.LB3 = HFAB(channel_num//2, large_kernel=31, split_group=4)

        sla_patterns = ['star', 'cross', 'corner']
        self.SLA1 = SLA_A_Ext(channel_num//4*3, depth, bias, ffn_bias, window_size, 
                              extend_size, sla_patterns[0], with_pe, dropout)
        self.SLA2 = SLA_A_Ext(channel_num//2, depth, bias, ffn_bias, window_size,
                              extend_size, sla_patterns[1], with_pe, dropout)
        self.SLA3 = SLA_A_Ext(channel_num//4, depth, bias, ffn_bias, window_size,
                              extend_size, sla_patterns[2], with_pe, dropout)
        
        self.SGA1 = SGA_A_Sample(channel_num//4*3, depth, bias, ffn_bias, window_size,
                                  sga_modes[0], with_pe, dropout)
        self.SGA2 = SGA_A_Sample(channel_num//2, depth, bias, ffn_bias, window_size,
                                  sga_modes[1], with_pe, dropout)
        self.SGA3 = SGA_A_Sample(channel_num//4, depth, bias, ffn_bias, window_size,
                                  sga_modes[2], with_pe, dropout)
        
        # Channel Attention Block
        self.BlockC = TrBlockC(channel_num, depth, bias, ffn_bias, window_size, with_pe, dropout)
        
        # Pointwise convolutions
        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num//4*3, channel_num//4*3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)

    def forward(self, x):
        # Stage 1: 4c -> 4c
        x = self.LB1(x)
        loc = x[:, :self.ch_split, :, :]  # c
        x, a1 = self.SLA1(x[:, self.ch_split:, :, :])  # 3c -> 3c
        x, a2 = self.SGA1(x)
        
        x = self.pw1(torch.cat((x, loc), 1))  # 4c
        x1 = x[:, :self.ch_split, :, :]  # c
        
        # Stage 2: 3c -> 3c
        x = self.LB2(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]  # c
        x, a3 = self.SLA2(x[:, self.ch_split:, :, :])  # 2c -> 2c
        x, a4 = self.SGA2(x)
        
        x = self.pw2(torch.cat((x, loc), 1))  # 3c
        x2 = x[:, :self.ch_split, :, :]  # c
        
        # Stage 3: 2c -> 2c
        x = self.LB3(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]  # c
        x, a5 = self.SLA3(x[:, self.ch_split:, :, :])  # c -> c
        x, a6 = self.SGA3(x)
        
        # Merge
        x = self.pw3(torch.cat((x1, x2, x, loc), 1))  # 4c
        x = self.BlockC(x)

        return x, a1, a2, a3, a4, a5, a6


class IDSA_Center_Block2(nn.Module):
    def __init__(self, channel_num=64, depth=0, bias=True, ffn_bias=True,
                 window_size=8, extend_size=4, sample_pattern='star',
                 sga_modes=None, with_pe=False, dropout=0.0):
        super(IDSA_Center_Block2, self).__init__()

        self.w = 8
        self.ch_split = channel_num // 4
        
        if sga_modes is None:
            sga_modes = ['center', 'row_mid', 'col_mid']
        
        # Local Blocks
        self.LB1 = HFAB(channel_num, large_kernel=31, split_group=4)
        self.LB2 = HFAB(channel_num//4*3, large_kernel=31, split_group=4)
        self.LB3 = HFAB(channel_num//2, large_kernel=31, split_group=4)

        sla_patterns = ['star', 'cross', 'corner']
        self.SLA1 = SLA_B_Ext(channel_num//4*3, depth, bias, ffn_bias, window_size,
                              extend_size, sla_patterns[0], with_pe, dropout)
        self.SLA2 = SLA_B_Ext(channel_num//2, depth, bias, ffn_bias, window_size,
                              extend_size, sla_patterns[1], with_pe, dropout)
        self.SLA3 = SLA_B_Ext(channel_num//4, depth, bias, ffn_bias, window_size,
                              extend_size, sla_patterns[2], with_pe, dropout)
        
        self.SGA1 = SGA_B_Sample(channel_num//4*3, depth, bias, ffn_bias, window_size,
                                  sga_modes[0], with_pe, dropout)
        self.SGA2 = SGA_B_Sample(channel_num//2, depth, bias, ffn_bias, window_size,
                                  sga_modes[1], with_pe, dropout)
        self.SGA3 = SGA_B_Sample(channel_num//4, depth, bias, ffn_bias, window_size,
                                  sga_modes[2], with_pe, dropout)
        
        # Channel Attention Block
        self.BlockC = TrBlockC(channel_num, depth, bias, ffn_bias, window_size, with_pe, dropout)
        
        # Pointwise convolutions
        self.pw1 = nn.Conv2d(channel_num, channel_num, 1)
        self.pw2 = nn.Conv2d(channel_num//4*3, channel_num//4*3, 1)
        self.pw3 = nn.Conv2d(channel_num, channel_num, 1)

    def forward(self, x, a1, a2, a3, a4, a5, a6):
        # Stage 1: 4c -> 4c
        x = self.LB1(x)
        loc = x[:, :self.ch_split, :, :]  # c
        x = self.SLA1(x[:, self.ch_split:, :, :], a1)  # 3c -> 3c
        x = self.SGA1(x, a2)
        
        x = self.pw1(torch.cat((x, loc), 1))  # 4c
        x1 = x[:, :self.ch_split, :, :]  # c
        
        # Stage 2: 3c -> 3c
        x = self.LB2(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]  # c
        x = self.SLA2(x[:, self.ch_split:, :, :], a3)  # 2c -> 2c
        x = self.SGA2(x, a4)
        
        x = self.pw2(torch.cat((x, loc), 1))  # 3c
        x2 = x[:, :self.ch_split, :, :]  # c
        
        # Stage 3: 2c -> 2c
        x = self.LB3(x[:, self.ch_split:, :, :])
        loc = x[:, :self.ch_split, :, :]  # c
        x = self.SLA3(x[:, self.ch_split:, :, :], a5)  # c -> c
        x = self.SGA3(x, a6)
        
        # Merge
        x = self.pw3(torch.cat((x1, x2, x, loc), 1))  # 4c
        x = self.BlockC(x)

        return x
########################################


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class IDSG(nn.Module):
    def __init__(self, channel_num=64, bias=True, block_num=4, **kwargs):
        super(IDSG, self).__init__()

        ffn_bias = kwargs.get("ffn_bias", False)
        window_size = kwargs.get("window_size", 0)
        pe = kwargs.get("pe", False)

        self.residual_layer = IDSA_Center_Block2(channel_num,bias,ffn_bias=ffn_bias,window_size=window_size,with_pe=pe)
        self.res_end = nn.Conv2d(channel_num,channel_num,1,1,0,bias=bias)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)
        
    def forward(self, x, a1, a2, a3, a4, a5, a6):
        out = self.residual_layer(x, a1, a2, a3, a4, a5, a6)
        out = self.res_end(out)
        out = out + x
        return self.esa(out)


class IDSG_A(nn.Module):
    def __init__(self, channel_num=64, bias=True, block_num=4, **kwargs):
        super(IDSG_A, self).__init__()

        ffn_bias = kwargs.get('ffn_bias', False)
        window_size = kwargs.get('window_size', 8)
        pe = kwargs.get('pe', False)

        group_list = []
        for _ in range(block_num):
            temp_res = IDSA_Center_Block1(channel_num,bias,ffn_bias=ffn_bias,window_size=window_size,with_pe=pe)
            group_list.append(temp_res)
        self.res_end = nn.Conv2d(channel_num,channel_num, 1, 1, 0, bias=bias)
        self.residual_layer = nn.Sequential(*group_list)
        esa_channel = max(channel_num // 4, 16)
        self.esa = ESA(esa_channel, channel_num)
        
    def forward(self, x):
        out, a1, a2, a3, a4, a5, a6 = self.residual_layer(x)
        out = self.res_end(out)
        out = out + x
        return self.esa(out), a1, a2, a3, a4, a5, a6


class HAESR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, **kwargs):
        super(HAESR, self).__init__()

        res_num = kwargs["res_num"]
        up_scale = 4
        bias = kwargs["bias"]

        self.res_num = res_num
        self.block0 = IDSG_A(channel_num=num_feat, **kwargs)
        self.block1 = IDSG(channel_num=num_feat, **kwargs)
        self.block2 = IDSG(channel_num=num_feat, **kwargs)

        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        self.window_size = kwargs["window_size"]
        self.up_scale = up_scale
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual= self.input(x)
        out, a1, a2, a3, a4, a5, a6 = self.block0(residual)
        out = self.block1(out, a1, a2, a3, a4, a5, a6)
        out = self.block2(out, a1, a2, a3, a4, a5, a6)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)
        
        out = out[:, :, :H*self.up_scale, :W*self.up_scale]
        return out
