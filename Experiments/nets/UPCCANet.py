import Experiments.Config as config
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import checkpoint_seq
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple

from pytorch_wavelets import DWTForward
from .SHCTrans import ChannelTransformer



class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
                                    nn.Conv2d(in_ch*4, out_ch, kernel_size=1, stride=1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace=True),
                                    )
        #print("downwt初始化")
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x
class ChannelSELayer(torch.nn.Module):
    """
    Implements Squeeze and Excitation
    """

    def __init__(self, num_channels):
        """
        Initialization

        Args:
            num_channels (int): No of input channels
        """

        super(ChannelSELayer, self).__init__()

        self.gp_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.reduction_ratio = 8  # default reduction ratio

        num_channels_reduced = num_channels // self.reduction_ratio

        self.fc1 = torch.nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = torch.nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.act = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(num_channels)


    def forward(self, inp):

        batch_size, num_channels, H, W = inp.size()

        out = self.act(self.fc1(self.gp_avg_pool(inp).view(batch_size, num_channels)))
        out = self.sigmoid(self.fc2(out))

        out = torch.mul(inp, out.view(batch_size, num_channels, 1, 1))

        out = self.bn(out)
        out = self.act(out)

        return out

class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 分割通道，三个卷积和一个恒等层，
        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        # square_kernel_size, padding=square_kernel_size//输出特征图大小不变
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size,dilation=2, padding=square_kernel_size // 2+1, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        # print("InceptionDWConv2d_forward:")
        # print(x.shape)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)  # 在通道维度分割
        # print(x_id.shape, x_hw.shape, x_w.shape, x_h.shape)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class InceptionDWConvTranspose2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        # 分割通道，三个卷积和一个恒等层，
        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.ConvTranspose2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.ConvTranspose2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.ConvTranspose2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        #print("InceptionDWConvTranspose2d_forward:")
        #print(x.shape)
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)  # 在通道维度分割
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features#没给就是输入通道数
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class StarMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, int(mlp_ratio * dim), 1, with_bn=False)
        self.f2 = ConvBN(dim, int(mlp_ratio * dim), 1, with_bn=False)
        self.g = ConvBN(int(mlp_ratio * dim), dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x
class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=3, act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.mean((2, 3)) # global average pooling
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x



class MetaNeXtBlock_SE(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,#控制covmlp中，中间扩大的通道数
            act_layer=nn.GELU,
            ls_init_value=1e-6,#层级缩放（Layer Scale）的初始值，默认为 1e-6
            drop_path=0.,#随机深度率

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        #print(mlp_ratio)
        # self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.se=ChannelSELayer(dim)
        self.mlp = StarMlp(dim,mlp_ratio=2)
        #如果 ls_init_value 不为 None，则创建一个层级缩放参数 gamma，用于调整特征的尺度。（可学习的层级缩放权重，作用在每个通道上）
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        x = self.se(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x


class MetaNeXtBlock(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=nn.Identity,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,#控制covmlp中，中间扩大的通道数
            act_layer=nn.GELU,
            ls_init_value=1e-6,#层级缩放（Layer Scale）的初始值，默认为 1e-6
            drop_path=0.,#随机深度率

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        #print(mlp_ratio)
        # self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.mlp = StarMlp(dim,mlp_ratio=2)
        #如果 ls_init_value 不为 None，则创建一个层级缩放参数 gamma，用于调整特征的尺度。（可学习的层级缩放权重，作用在每个通道上）
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class MetaNeXtStage_blockwithse(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.depth=depth
        self.grad_checkpointing = False
        if ds_stride > 1:
            # self.downsample = nn.Sequential(
            #     norm_layer(in_chs),
            #     nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            # )
            # self.downsample = nn.Sequential(
            #     norm_layer(in_chs),
            #     Down_wt(in_chs, out_chs)
            # )
            self.downsample = Down_wt(in_chs,out_chs)
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock_SE(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        #print("in_chs:", in_chs, "out_chs", out_chs, "depth", depth)
    def forward(self, x):
        #print("MetaNeXtStage_forward:",self.depth)
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

class MetaNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.depth=depth
        self.grad_checkpointing = False
        if ds_stride > 1:
            # self.downsample = nn.Sequential(
            #     norm_layer(in_chs),
            #     nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            # )
            # self.downsample = nn.Sequential(
            #     norm_layer(in_chs),
            #     Down_wt(in_chs, out_chs)
            # )
            self.downsample = Down_wt(in_chs,out_chs)
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        #print("in_chs:", in_chs, "out_chs", out_chs, "depth", depth)
    def forward(self, x):
        #print("MetaNeXtStage_forward:",self.depth)
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

class MetaNeXtStage_Conv(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        self.depth=depth
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
            # self.downsample = nn.Sequential(
            #     norm_layer(in_chs),
            #     Down_wt(in_chs, out_chs)
            # )
            # self.downsample = Down_wt(in_chs,out_chs)
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        #print("in_chs:", in_chs, "out_chs", out_chs, "depth", depth)
    def forward(self, x):
        #print("MetaNeXtStage_forward:",self.depth)
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1).contiguous()
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class MetaNeXtUpStage0(nn.Module):
    def __init__(
            self,
            i,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        """调试"""
        self.i=i
        """调试"""

        self.depth=depth
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.upsample = nn.Sequential(
                norm_layer(in_chs),
                nn.ConvTranspose2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
            )
        else:
            self.upsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        #print("in_chs:",in_chs,"out_chs",out_chs,"depth",depth)
    def forward(self, x):
        #print("MetaNeXtUpStage_forward:", self.i,self.depth)
        #print(x.device)
        x = self.upsample(x)
        #print(x.device)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x
#动态上采样
class MetaNeXtUpStage(nn.Module):
    def __init__(
            self,
            i,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        """调试"""
        self.i=i
        """调试"""

        self.depth=depth
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.upsample = nn.Sequential(
                norm_layer(in_chs),
                # nn.ConvTranspose2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
                DySample(in_channels=in_chs),
                nn.Conv2d(in_chs,out_chs,kernel_size=1)
            )
        else:
            self.upsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        #print("in_chs:",in_chs,"out_chs",out_chs,"depth",depth)
    def forward(self, x):
        #print("MetaNeXtUpStage_forward:", self.i,self.depth)
        #print(x.device)
        x = self.upsample(x)
        #print(x.device)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class MetaNeXtUpStage_blockwithse(nn.Module):
    def __init__(
            self,
            i,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        """调试"""
        self.i=i
        """调试"""

        self.depth=depth
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.upsample = nn.Sequential(
                norm_layer(in_chs),
                # nn.ConvTranspose2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
                DySample(in_channels=in_chs),
                nn.Conv2d(in_chs,out_chs,kernel_size=1)
            )
        else:
            self.upsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock_SE(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        #print("in_chs:",in_chs,"out_chs",out_chs,"depth",depth)
    def forward(self, x):
        #print("MetaNeXtUpStage_forward:", self.i,self.depth)
        #print(x.device)
        x = self.upsample(x)
        #print(x.device)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x

class MetaNeXtUpStage_ConvTranspose(nn.Module):
    def __init__(
            self,
            i,
            in_chs,
            out_chs,
            ds_stride=2,
            depth=2,
            drop_path_rates=None,
            ls_init_value=1.0,
            token_mixer=nn.Identity,
            act_layer=nn.GELU,
            norm_layer=None,
            mlp_ratio=4,
    ):
        super().__init__()
        """调试"""
        self.i=i
        """调试"""

        self.depth=depth
        self.grad_checkpointing = False
        if ds_stride > 1:
            self.upsample = nn.Sequential(
                norm_layer(in_chs),
                nn.ConvTranspose2d(in_chs, out_chs, kernel_size=ds_stride, stride=ds_stride),
                # DySample(in_channels=in_chs),
                # nn.Conv2d(in_chs,out_chs,kernel_size=1)
            )
        else:
            self.upsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(MetaNeXtBlock(
                dim=out_chs,
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixer,
                act_layer=act_layer,
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratio,
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)
        #print("in_chs:",in_chs,"out_chs",out_chs,"depth",depth)
    def forward(self, x):
        #print("MetaNeXtUpStage_forward:", self.i,self.depth)
        #print(x.device)
        x = self.upsample(x)
        #print(x.device)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


#全的
class UPCCANet(nn.Module):
    r""" MetaNeXt
        A PyTorch impl of : `InceptionNeXt: When Inception Meets ConvNeXt`  - https://arxiv.org/pdf/2203.xxxxx.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 9, 3)
        dims (tuple(int)): Feature dimension at each stage. Default: (96, 192, 384, 768)
        token_mixers: Token mixer function. Default: nn.Identity
        norm_layer: Normalziation layer. Default: nn.BatchNorm2d
        act_layer: Activation function for MLP. Default: nn.GELU
        mlp_ratios (int or tuple(int)): MLP ratios. Default: (4, 4, 4, 3)
        head_fn: classifier head
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            config,
            in_chans=3,
            num_classes=1000,
            depths=(3, 3, 9, 3),
            img_size=224,
            dims=(96, 192, 384, 768),
            token_mixers=InceptionDWConv2d,
            up_token_mixers=InceptionDWConvTranspose2d,
            norm_layer=LayerNorm2d,
            act_layer=nn.GELU,
            mlp_ratios=(4, 4, 4, 3),
            head_fn=MlpHead,
            drop_rate=0.,
            drop_path_rate=0.,
            ls_init_value=1e-6,
            **kwargs,
    ):
        super().__init__()

        num_stage = len(depths)
        self.num_stage = num_stage
        # 如果不是列表或元组
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage
            up_token_mixers = [up_token_mixers] * num_stage
        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )

        # 计算每个阶段的dropout路径率。
        # torch.linspace(start, end, steps)：这是一个PyTorch函数，用于在指定的开始值start和结束值end之间生成steps个点，形成一个线性分布。
        """
        start=0：分布的起始值为0。
        end=drop_path_rate：分布的结束值为构造函数参数drop_path_rate，表示随机丢弃路径的最大比例。
        steps=sum(depths)：点的数量为depths元组中所有值的和，即模型中所有阶段的块数总和。
        .split(depths)：这是PyTorch中Tensor的split方法，用于将一个张量分割成多个小块，每块的大小由depths列表中的值决定。这里的分割是沿着linspace生成的一维张量进行的。
        列表推导式，用于将上一步得到的每个分割后的张量（Tensor对象）转换为Python列表（list对象）。这是因为drop_path_rates需要是一个列表的列表，其中每个内部列表对应一个阶段的丢弃路径率。
        """
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        self.downstages = nn.ModuleList()
        prev_chs = dims[0]

        # SE
        self.selist = nn.ModuleList()
        for i in range(num_stage - 1):
            seblock = ChannelSELayer(dims[i])
            self.selist.append(seblock)

        self.chan_trans=ChannelTransformer(config,img_size=img_size//4,channel_num=dims,)
        # feature resolution stages, each consisting of multiple residual blocks
        for i in range(num_stage):
            out_chs = dims[i]
            self.downstages.append(MetaNeXtStage(
                prev_chs,
                out_chs,
                ds_stride=2 if i > 0 else 1,  # 因为下采样在每个block内前部执行，经过stem后通道数已经是96，第一层不需要下采样
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=token_mixers[i],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[i],
            ))
            prev_chs = out_chs

        self.upstages = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        # upstage
        for i in range(num_stage):
            out_chs = dims[num_stage - i - 1]

            concat_linear = nn.Conv2d(2 * out_chs, out_chs, kernel_size=1, stride=1, padding=0)
            self.concat_back_dim.append(concat_linear)
            if i == 0:
                continue

            self.upstages.append(MetaNeXtUpStage(
                num_stage - i - 1,
                prev_chs,
                out_chs,
                ds_stride=2,  # 因为下采样在每个block内前部执行，最后一层最后本来就没有上采样，而且有图stage4公用，第一次上采样不需要像第一次下采样特殊判断用恒等层
                depth=depths[num_stage - i - 1],
                drop_path_rates=dp_rates[num_stage - i - 1],
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                token_mixer=up_token_mixers[num_stage - i - 1],
                norm_layer=norm_layer,
                mlp_ratio=mlp_ratios[num_stage - i - 1],
            ))

            prev_chs = out_chs

        self.upx4 = nn.Sequential(
            nn.ConvTranspose2d(dims[0], dims[0], kernel_size=4, stride=4),
            norm_layer(dims[0])
        )
        self.output = nn.Conv2d(in_channels=dims[0], out_channels=self.num_classes, kernel_size=1, bias=False)

        # 根据类别数初始化最后的激活函数
        if self.num_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = nn.Identity()
        # self.num_features = prev_chs
        # self.head = head_fn(self.num_features, num_classes, drop=drop_rate)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.downstages:
            s.grad_checkpointing = enable
        for s in self.upstages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_down_features(self, x):
        x = self.stem(x)
        # print("after_stem",x.shape)
        x_downsample = []
        for stage in self.downstages:
            x = stage(x)
            x_downsample.append(x)
        return x, x_downsample

    def forward_up_features(self, x, x_downsample):
        x1, x2, x3, x4 = x_downsample
        o1, o2, o3 = self.chan_trans(x1, x2, x3)
        olist = [o1, o2, o3]
        for inx, stage in enumerate(self.upstages):
            if inx == 0:
                x = stage(x)
            else:

                o=olist[self.num_stage-1-inx]
                o = self.selist[self.num_stage - 1 - inx](o)
                x = torch.cat([x,o], dim=1)

                x = self.concat_back_dim[inx](x)

                x = stage(x)
        o = olist[0]
        o = self.selist[0](o)
        x = torch.cat([x, o], dim=1)
        x = self.concat_back_dim[3](x)
        x = self.upx4(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        x, x_downsample = self.forward_down_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.output(x)
        x = self.last_activation(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchinfo import summary
    from Experiments import Config as config
    from thop import profile
    from Experiments.test_model import throughput_count
    from SHCTrans import ChannelTransformer

    sha_config=config.get_SHA_config()
    net =UPCCANet(sha_config,depths=(2, 2, 2, 2), dims=(64, 128, 256, 512),
                              in_chans=config.n_channels,num_classes=config.n_labels,
                              ).cuda()


    input_data=torch.randn(1,3,224,224).cuda()

    pred=net(input_data)
    print(pred.shape)
    print(summary(net,input_size=(1,3,224,224),depth=5))
    # input_data = torch.randn(1, 3, 224, 224).cuda()
    #
    flops, params = profile(net, inputs=(input_data,))
    print(f"GFLOPs: {flops / 1e+9}, Params(M): {params / 1e+6}\n")
    throughput = throughput_count(net)
    print(f"Throughput(images/s): {throughput}")