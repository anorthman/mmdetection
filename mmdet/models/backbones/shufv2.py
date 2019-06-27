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
class ChannelShuffle(nn.Module):
  def __init__(self, group=1):
    assert group > 1
    super(ChannelShuffle, self).__init__()
    self.group = group
  def forward(self, x):
    """https://github.com/Randl/ShuffleNetV2-pytorch/blob/master/model.py
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % self.group == 0)
    channels_per_group = num_channels // self.group
    # reshape
    x = x.view(batchsize, self.group, channels_per_group, height, width)
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

class Shufv2Unit(nn.Module):
    def __init__(self, _in, _out, kernel_size=3, padding=1, 
                        stride=1, c_tag=0.5, activation=nn.ReLU,
                         SE=False, residual=False, groups=2):

        super(Shufv2Unit, self).__init__()
        self.stride = stride
        self._in = _in
        self._out = _out
        self.groups = groups
        self.activation = activation(inplace=True)
        self.channel_shuffle = ChannelShuffle(group=groups)
        if self.stride == 1:
            assert _out == _in                    
            self.left_part = round(c_tag * _in)
            self.right_part_in = _in - self.left_part
            self.right_part_out = _out - self.left_part
            self.conv1 = nn.Conv2d(self.right_part_in, self.right_part_out, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.right_part_out)
            self.conv2 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=kernel_size, padding=padding, bias=False,
                                   groups=self.right_part_out)
            self.bn2 = nn.BatchNorm2d(self.right_part_out)
            self.conv3 = nn.Conv2d(self.right_part_out, self.right_part_out, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(self.right_part_out)
        elif self.stride == 2:
            assert _out == _in * 2
            self.conv1r = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
            self.bn1r = nn.BatchNorm2d(_in)
            self.conv2r = nn.Conv2d(_in, _in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=_in)
            self.bn2r = nn.BatchNorm2d(_in)
            self.conv3r = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
            self.bn3r = nn.BatchNorm2d(_in)

            self.conv1l = nn.Conv2d(_in, _in, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=_in)
            self.bn1l = nn.BatchNorm2d(_in)
            self.conv2l = nn.Conv2d(_in, _in, kernel_size=1, bias=False)
            self.bn2l = nn.BatchNorm2d(_in)
        else:
            raise ValueError   

    def forward(self, x):
        if self.stride == 1:
            left = x[:, :self.left_part, :, :]
            right = x[:, self.left_part:, :, :]
            out = self.conv1(right)
            out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            out = self.bn2(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.activation(out)

            out = self.channel_shuffle(torch.cat((left, out), 1))
        elif self.stride == 2:
            out_r = self.conv1r(x)
            out_r = self.bn1r(out_r)
            out_r = self.activation(out_r)

            out_r = self.conv2r(out_r)
            out_r = self.bn2r(out_r)
            out_r = self.conv3r(out_r)
            out_r = self.bn3r(out_r)
            out_r = self.activation(out_r)

            out_l = self.conv1l(x)
            out_l = self.bn1l(out_l)
            out_l = self.conv2l(out_l)
            out_l = self.bn2l(out_l)
            out_l = self.activation(out_l)
            out = self.channel_shuffle(torch.cat((out_r, out_l), 1))
        else:
            raise ValueError            

        return out
        

@BACKBONES.register_module
class Shufv2(nn.Module):
    """Shufv2 backbone.

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
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 normalize=dict(type='BN', frozen=False)):
        super(Shufv2, self).__init__()
        self.depth = depth
        self.num_stages = len(self.depth)
        assert num_stages > 0
        self.strides = strides
        self.dilations = dilations
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
                    _ops.append(Shufv2Unit(_in, _out, kernel_size=self.kernel_size[k], 
                            padding=int(self.kernel_size[k]/2), stride=2))
                else:      
                    _ops.append(Shufv2Unit(_in, _in, kernel_size=self.kernel_size[k], 
                            padding=int(self.kernel_size[k]/2), stride=1))
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
