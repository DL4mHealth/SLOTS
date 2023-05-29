import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            # Cai: :-self.remove 除了最后一个取全部
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        # Cai: 如果输入不等于输出或final，self.projector有值，否则self.projector为None
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        # Cai: When the approximate argument is ‘none’, it applies element-wise the function \text{GELU}(x) = x * \Phi(x)GELU(x)=x∗Φ(x)
        # Cai: 非线性激活函数gelu
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    # Cai: DilatedConvEncoder(in_channels=64, channesl=[64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 320], kernel_size=3) len(channels)=11
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
            # i=0, ConvBlock(64, 64, kernel_size=3, dilation=1, final=False) i=1, ConvBlock(64, 64, kernel_size=3, dilation=2,final=False)
        ])

    def forward(self, x):
        return self.net(x)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        #x=x.view(x.size(0),-1)
        return x
"""
x = x.view(x.size()[0], -1) 这句话的出现就是为了将前面操作输出的多维度的tensor展平成一维，
然后输入分类器，-1是自适应分配，指在不知道函数有多少列的情况下，根据原tensor数据自动分配列数。
"""
class ECNN(nn.Module):
    def __init__(self, nb_classes, include_top=True, weights=True):
        super(ECNN,self).__init__()
        self.in_planes = 64
        # stem的网络层
        self.conv1 = nn.Conv2d(1, 64 , kernel_size=(3,3), stride=(1,1), padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # 定义
        self.block1 = self.make_layers(64, 128)
        self.block2 = self.make_layers(128, 256)
        self.block3 = self.make_layers(256, 512)
        self.flatten= Flatten()
        # self.linear1 = nn.Linear(14336,128)
        self.linear1 = nn.Linear(32768, 128)
        # self.linear2 = nn.Linear(2048, nb_classes)
        self.include_top=include_top
        self.weights=weights

    # 定义 激活函数+BN
    def make_post(self, ch_in):
        layers = [
            nn.LeakyReLU(0.3),
            nn.BatchNorm2d(ch_in, momentum=0.99)
        ]
        return nn.Sequential(*layers)

    # 定义核心模块 空间可分离卷积 深度可分离卷积 最大池化  作用:降低计算量
    def make_layers(self, ch_in, ch_out):
        layers = [
            # ch_in=32,  (3,32)
            nn.Conv2d(1, ch_in, kernel_size=(1, 1), stride=(1, 1), bias=False, padding=0, dilation=(1, 1))
            if ch_in == 32
            else nn.Conv2d(ch_in, ch_in, kernel_size=(1, 1), stride=(1, 1), bias=False, padding=0,dilation=(1, 1)),
            self.make_post(ch_in),
            #DW
            nn.Conv2d(ch_in, 1*ch_in, groups=ch_in, kernel_size=(1, 3), padding=(0, 1), bias=False,dilation=(1,1)),
            self.make_post(ch_in),
            nn.MaxPool2d(kernel_size=(2,1),stride=(2,1)),
            #DW
            nn.Conv2d(ch_in, 1*ch_in, groups=ch_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False, dilation = (1, 1)),
            self.make_post(ch_in),
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, dilation=(1, 1)),
            self.make_post(ch_out)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # 判断有没有全连接
        if self.include_top:
            x = self.flatten(x)
            x = self.linear1(x)
            # x = self.linear2(x)
        return x
