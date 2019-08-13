import warnings

import torch.nn as nn
from mmcv.cnn import kaiming_init, constant_init
from mmcv.cnn import xavier_init


from .norm import build_norm_layer


class ConvSet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 normalize=None,
                 activation='leakyrelu',
                 inplace=True,
                 upsample=False,
                 output=False):
        super(ConvSet, self).__init__()
        self.with_norm = normalize is not None
        self.with_activatation = activation is not None
        self.with_bias = bias
        self.activation = activation
        self.upsample = upsample
        self.output = output

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        if self.upsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
            # self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        elif self.output:
            self.conv1 = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=padding, bias=bias)
            self.conv_out = nn.Conv2d(in_channels, 75, kernel_size=1, stride=1, bias=bias)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[0], stride, padding, dilation, groups, bias=bias)
            self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size[1], stride, padding, dilation, groups, bias=bias)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size[2], stride, padding, dilation, groups, bias=bias)
            self.conv4 = nn.Conv2d(out_channels, in_channels, kernel_size[3], stride, padding, dilation, groups, bias=bias)
            self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size[4], stride, padding, dilation, groups, bias=bias)

        self.in_channels = self.conv1.in_channels
        self.out_channels = self.conv1.out_channels
        self.kernel_size = self.conv1.kernel_size
        self.stride = self.conv1.stride
        self.padding = self.conv1.padding
        self.dilation = self.conv1.dilation
        self.transposed = self.conv1.transposed
        self.output_padding = self.conv1.output_padding
        self.groups = self.conv1.groups

        if self.with_norm:
            if self.output:
                self.norm1_name, norm1 = build_norm_layer(normalize, in_channels, postfix=1)
                self.add_module(self.norm1_name, norm1)
            elif self.upsample:
                self.norm1_name, norm1 = build_norm_layer(normalize, out_channels, postfix=1)
                self.add_module(self.norm1_name, norm1)
            else:
                self.norm1_name, norm1 = build_norm_layer(normalize, out_channels, postfix=1)
                self.add_module(self.norm1_name, norm1)
                self.norm2_name, norm2 = build_norm_layer(normalize, in_channels, postfix=2)
                self.add_module(self.norm2_name, norm2)
                self.norm3_name, norm3 = build_norm_layer(normalize, out_channels, postfix=3)
                self.add_module(self.norm3_name, norm3)
                self.norm4_name, norm4 = build_norm_layer(normalize, in_channels, postfix=4)
                self.add_module(self.norm4_name, norm4)
                self.norm5_name, norm5 = build_norm_layer(normalize, out_channels, postfix=5)
                self.add_module(self.norm5_name, norm5)

        if self.with_activatation:
            assert activation in ['leakyrelu'], 'Only ReLU supported.'
            if self.activation == 'leakyrelu':
                self.activate = nn.LeakyReLU(negative_slope=0.1, inplace=inplace)

        # Default using msra init
        self.init_weights()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    @property
    def norm4(self):
        return getattr(self, self.norm4_name)

    @property
    def norm5(self):
        return getattr(self, self.norm5_name)

    def init_weights(self):
        if self.output:
            xavier_init(self.conv1, distribution='uniform')
            xavier_init(self.conv_out, distribution='uniform')
            constant_init(self.norm1, 1, bias=0)
        elif self.upsample:
            xavier_init(self.conv1, distribution='uniform')
            # xavier_init(self.conv2, distribution='uniform')
            constant_init(self.norm1, 1, bias=0)
        else:
            for i in range(1, 6):
                conv = getattr(self, 'conv{}'.format(i))
                xavier_init(conv, distribution='uniform')
                if self.with_norm:
                    norm = getattr(self, 'norm{}'.format(i))
                    constant_init(norm, 1, bias=0)

    def forward(self, x, activate=True, norm=True):
        if self.upsample:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.activate(x)
            # x = self.conv2(x)
        elif self.output:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.activate(x)
            x = self.conv_out(x)
        else:
            for i in range(1, 6):
                conv = getattr(self, 'conv{}'.format(i))
                x = conv(x)
                if norm and self.with_norm:
                    norm = getattr(self, 'norm{}'.format(i))
                    x = norm(x)
                if activate and self.with_activatation:
                    x = self.activate(x)
        return x
