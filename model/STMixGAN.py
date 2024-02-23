import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from timm.models.layers import DropPath
from timm.models.mlp_mixer import MixerBlock


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super(PatchEmbed, self).__init__()

        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)

        if pad_input:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchMerging, self).__init__()

        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, W % 2, 0, H % 2, 0, 0))
        x0 = x[:, 0::2, 0::2, :] 
        x1 = x[:, 1::2, 0::2, :] 
        x2 = x[:, 0::2, 1::2, :] 
        x3 = x[:, 1::2, 1::2, :] 
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, act=nn.GELU, drop=0.):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super(WindowAttention, self).__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_positive_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
      
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_positive_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_positive_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_window = mask.shape[0]
            attn = attn.view(B_ // num_window, num_window, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0., mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim=dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = (int(dim * mlp_ratio))
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, H, W, C)

        x_r = (self.window_size - W % self.window_size) % self.window_size
        x_d = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, x_r, 0, x_d))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0.:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if x_r > 0 or x_d > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.norm1(self.drop_path(x))
        x = x + self.norm2(self.drop_path(self.mlp(x)))

        return x


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )


def window_reverse(windows, window_size: int, H: int, W:int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super(BasicLayer, self).__init__()

        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size, shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        H_padding = int(np.ceil(H / self.window_size)) * self.window_size
        W_padding = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, H_padding, W_padding, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size),
                    slice(-self.window_size, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0, 0.0
        )

        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)
            y = x
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return y, x, H, W


class SwinTransformer(nn.Module):
    def __init__(self, downsapmle_size=1, in_channels=4, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=4, mlp_ratio=4.,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, **kwargs):
        super(SwinTransformer, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(patch_size=downsapmle_size, in_channels=in_channels, embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], norm_layer=norm_layer, downsample=PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layers)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        extract_layers = []
        for layer in self.layers:
            y, x, H, W = layer(x, H, W)
            extract_layers.append(y)
        return extract_layers

class MLP_Mixer(MixerBlock):
    def __init__(self, dim, input_resolution=None, mlp_ratio=4., drop=0., drop_path=0.1):
        seq_len = input_resolution[0] * input_resolution[1]
        super().__init__(dim, seq_len=seq_len,
                         mlp_ratio=(0.5, mlp_ratio), drop_path=drop_path, drop=drop)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class Down(nn.Module):
    def __init__(self, in_channels, kernel_size, size):
        super(Down, self).__init__()

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        )
       
    def forward(self, x):
        return self.branch_pool(x)


class Up(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv(self.up(x))

class LFEU(nn.Module):
    def __init__(self, in_channels, out_channels, size, kernel_size=3):
        super(LFEU, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=kernel_size//2),
            nn.LayerNorm([out_channels, size, size]),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=kernel_size, padding=kernel_size//2),
            nn.LayerNorm([out_channels, size, size]),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Out, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size, padding=1),
            nn.Conv2d(2, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return self.conv(x)
        

class STMixNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super().__init__()

        self.transformer = SwinTransformer(in_channels=in_channels, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
        self.ecblock1 = LFEU(in_channels, 96, 128)
        self.down1 = Down(96, 2, 64)
        self.ecblock2 = LFEU(96, 192, 64)
        self.down2 = Down(192, 2, 32)
        self.ecblock3 = LFEU(192, 384, 32)
        self.down3 = Down(384, 2, 16)
        self.ecblock4 = LFEU(384, 768, 16)

        self.up2 = Up(768, 2)
        self.dcblock6 = LFEU(1152, 384, 32)
        self.up3 = Up(384, 2)
        self.dcblock7 = LFEU(576, 192, 64)
        self.up4 = Up(192, 2)
        self.dcblock8 = LFEU(288, 96, 128)

        self.mix0 = MLP_Mixer(96, [128, 128])
        self.mix1 = MLP_Mixer(192, [64, 64])
        self.mix2 = MLP_Mixer(384, [32, 32])
        self.mix3 = MLP_Mixer(768, [16, 16])

        self.out = Out(96, out_channels) 

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        b, t, c, h, w = x.shape
        x = torch.reshape(x, [b, t*c, h, w])
        x = x.view(b, t*c, h, w)

        y = x
        y = self.transformer(y)
        x, y1s, y2s, y3s, y4s = x, *y
        y1s = y1s.transpose(-1, -2).reshape(-1, 96, 128, 128)
        y2s = y2s.transpose(-1, -2).reshape(-1, 192, 64, 64)
        y3s = y3s.transpose(-1, -2).reshape(-1, 384, 32, 32)
        y4s = y4s.transpose(-1, -2).reshape(-1, 768, 16, 16)

        x1s = self.ecblock1(x)
        x1s = x1s + y1s

        x2s = self.ecblock2(self.down1(x1s))
        x2s = x2s + y2s

        x3s = self.ecblock3(self.down2(x2s))
        x3s = x3s + y3s

        x4s = self.ecblock4(self.down3(x3s))
        x4s = x4s + y4s

        x = x4s + self.mix3(x4s)
        x = torch.cat((self.up2(x), x3s+self.mix2(x3s)), dim=1) #
        x = self.dcblock6(x)
        x = torch.cat((self.up3(x), x2s+self.mix1(x2s)), dim=1) #
        x = self.dcblock7(x)
        x = torch.cat((self.up4(x), x1s +self.mix0(x1s)), dim=1) #
        x = self.dcblock8(x)

        return self.out(x)
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_planes=256):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class DCNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.cbam = CBAM(256)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 1)
        )
        
    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.cbam(x)
        x = self.conv2(x)
        x = self.linear(x)
        return x
