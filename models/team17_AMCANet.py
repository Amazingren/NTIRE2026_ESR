import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
#from basicsr.utils.registry import ARCH_REGISTRY


class ConvAttention(nn.Module):
    def __init__(self,in_c,out_c,idx):
        super(ConvAttention, self).__init__()
        self.idx = idx
        self.stride = 4*self.idx # 1 4 8 12 16 16
        if self.stride > 16:
            self.stride = 16

        self.conv1x1 = nn.Conv2d(in_c,out_c * 2,1,1,0)
        
        self.conv3x3_1 = nn.Conv2d(out_c,out_c,3,1,1,groups=out_c)
        self.conv3x3_2 = nn.Conv2d(out_c,out_c,3,1,1,groups=out_c)
        self.conv3x3_3 = nn.Conv2d(out_c,out_c,3,1,1,groups=out_c)

        self.conv3x3_v = nn.Conv2d(out_c,out_c,3,1,1,groups=out_c)

        self.conv1x1_2 = nn.Conv2d(out_c,out_c,1,1,0)
        self.conv1x1_3 = nn.Conv2d(out_c,out_c,1,1,0)

        
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        input = self.conv1x1(x) # 降低channel

        qk, v = torch.chunk(input, 2, dim=1) # 分成两部分

        qk_res = self.conv3x3_1(qk)
        if self.idx > 0:
            qk = F.max_pool2d(qk_res, kernel_size=2*self.stride-1, stride=self.stride) # 降分辨率
        else:
            qk = qk_res

        qk = self.conv3x3_2(qk)
        #qk = F.gelu(qk)
        qk = self.conv3x3_3(qk)
        qk = self.conv1x1_2(qk)
        if self.idx > 0:
            qk = F.interpolate(qk, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) # bilinear,nearest
        qk = qk + qk_res # 残差
        qk = self.conv1x1_3(qk)
        qk = self.sigmoid(qk)

        v = self.conv3x3_v(v)

        out = qk * v # 注意力加权

        return out

class FSG(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)//2 * 2
        self.conv1x1_1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)

        self.conv3x3_2 = nn.Conv2d(hidden_dim//2,hidden_dim//2,3,1,1,groups=hidden_dim//2)
        self.conv1x1_2 = nn.Conv2d(hidden_dim//2,hidden_dim//2,1,1,0)

        self.conv1x1_3 = nn.Conv2d(hidden_dim//2, dim, 1, 1, 0)

    def forward(self, x):
        x1, x2 = self.conv1x1_1(x).chunk(2, dim=1)
        out = self.conv1x1_3(torch.sigmoid(self.conv1x1_2(self.conv3x3_2(x1))) * x2)
        return out

class AMCA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(AMCA, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.heads = nn.ModuleList([ConvAttention(dim,head_dim,n) for n in range(num_heads)])
        self.fusion_conv = nn.Conv2d(head_dim*num_heads, dim, 1, 1, 0)
        
    def forward(self, x):
        head_outs = [head(x) for head in self.heads]  # 每个头的输出
        out = torch.cat(head_outs, dim=1)  # 拼接所有头的输出
        out = self.fusion_conv(out)  # 融合
        return out

class MHCB(nn.Module):
    def __init__(self, dim, num_heads=4):
        super(MHCB, self).__init__()
        self.amca = AMCA(dim,num_heads)
        self.fsg = FSG(dim, growth_rate=2.0)
        
    def forward(self, x):
        out = self.amca(F.normalize(x)) + x
        out = self.fsg(F.normalize(out)) + out  # FFN
        return out


#@ARCH_REGISTRY.register()
class AMCANet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, dim=32, n_blocks=4, upscaling_factor=4, num_heads=4):
        super(AMCANet, self).__init__()
        self.n_blocks = n_blocks
        self.num_heads = num_heads
        
        self.head = nn.Conv2d(in_nc, dim, 3, 1, 1, bias=True)
        self.blocks = nn.ModuleList([MHCB(dim, num_heads) for _ in range(n_blocks)])
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        self.tail = nn.Conv2d(dim, out_nc * upscaling_factor * upscaling_factor, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscaling_factor)
        self.cuda()(torch.randn(1, 3, 256, 256).cuda())

    def forward(self, x):
        fea = self.head(x)
        out = fea
        
        for i, block in enumerate(self.blocks):
            fea = block(fea)  # 获取每个块的输出和头的输出
        
        out = self.conv1x1(fea) + out
        
        out = self.pixel_shuffle(self.tail(out))
        
        # 返回最终输出
        return out
