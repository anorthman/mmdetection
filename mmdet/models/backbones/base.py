import logging

import torch.nn as nn
import torch.utils.checkpoint as cp

from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_norm_layer
from itertools import accumulate
import torch

# @BACKBONES.register_module        
class BaseConv(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 kernel_size=3,
                 group=1,
                 padding=1,
                 normalize=dict(type='BN'),
                 activate=nn.ReLU):
        super(BaseConv, self).__init__()
        self.stride = stride
        self.dilation = dilation
        self.kernel_size =kernel_size
        self.padding = padding
        self.group = group
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=self.kernel_size,stride=self.stride, 
                        padding=self.padding,groups=self.group,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if activate == nn.PReLU:
            self.activate = activate(planes)
        else:	
            self.activate = activate(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)

        return x

@BACKBONES.register_module
class Base(nn.Module):
    """Base backbone.

    Args: 
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    def __init__(self,
                 base,
                 depth,
                 inchn,
                 kernel_size=[3],
                 group=[1],
                 num_stages=4,
                 activate=nn.ReLU,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 normalize=dict(type='BN', frozen=False)):
        super(Base, self).__init__()
        self.depth = depth
        self.num_stages = len(self.depth)
        assert num_stages > 0
        self.strides = strides
        self.dilations = dilations
        self.activate = activate
        assert self.num_stages == len(self.strides) == len(self.dilations)
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.out = [list(accumulate(self.depth))[z]-1 for z in out_indices]
        # self.normalize = normalize
        self.kernel_size = kernel_size
        self.depth_num = sum(self.depth)
        self.base = nn.Sequential(*base) 
        self._in = inchn
        if len(kernel_size) == 1:
            self.kernel_size = kernel_size * self.depth_num
        elif len(kernel_size) == self.depth_num:
            self.kernel_size = kernel_size
        else:
            raise ValueError("the length of kernel_size must equals 1 or sum(depths)")
        if len(group) == 1:
            self.group = group * self.depth_num
        elif len(group) == self.depth_num:
            self.group = group
        else:
            raise ValueError("the length of group must equals 1 or sum(depths)")
        self._ops = self.build()


    def build(self):
        _ops = nn.ModuleList()
        k = 0
        _out = self._in
        for i in range(len(self.depth)):
            for j in range(self.depth[i]):
                _in = _out
                if j == 0 and self.strides[i] == 2: 
                    _out = _out * 2       
                    _ops.append(BaseConv(_in, _out, kernel_size=self.kernel_size[k], group=self.group[k],
                            padding=int(self.kernel_size[k]/2), stride=2,activate=self.activate))
                else:      
                    _ops.append(BaseConv(_in, _in, kernel_size=self.kernel_size[k], group=self.group[k],
                            padding=int(self.kernel_size[k]/2), stride=1,activate=self.activate))
                k += 1       
        assert len(_ops) == self.depth_num
        return _ops

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.base(x)
        outs = []
        for i in range(len(self._ops)):
            x = self._ops[i](x)
            if i in self.out:
                outs.append(x)
        if len(self.out) == 1:
            return outs[0]
        else:
            return tuple(outs) 
