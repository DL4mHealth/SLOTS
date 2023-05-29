import torch.nn.functional as F
import torch.nn as nn

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
    def __init__(self, include_top=True, weights=True):
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
            nn.MaxPool2d(kernel_size=(1,1),stride=(1,1)),
            #DW
            nn.Conv2d(ch_in, ch_out, groups=ch_in, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0), bias=False, dilation = (1, 1)),
            self.make_post(ch_out),
            # nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1), bias=False, padding=0, dilation=(1, 1)),
            # nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=(1, 1), padding=1, bias=False),
            # nn.Conv2d(ch_in, ch_out, kernel_size=(1, 2), stride=(1, 2), bias=False, dilation=(1, 1)),

            # self.make_post(ch_out)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        cc=1*x
        x = F.relu(self.bn1(self.conv1(x)))
        cc1 = 1 * x
        x = self.block1(x)
        cc2 = 1 * x
        x = self.block2(x)
        cc3 = 1 * x
        x = self.block3(x)
        cc4 = 1 * x
        # 判断有没有全连接
        if self.include_top:
            x = self.flatten(x)
            x = self.linear1(x)
            # x = self.linear2(x)
        ee=1*x
        return x
