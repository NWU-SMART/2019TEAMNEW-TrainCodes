# ------------------------1、导入需要的模块-------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from torch.nn.modules.utils import _triple


# ------------------------1、导入需要的模块-------------------------

# ------------------------2、定义时空卷积 ST_Block（3DCNN）-------------------
class SpatioTemporalConv(nn.Module):
    # 初始化参数（输入通道、输出通道、卷积核大小、步长、填充）
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # 用  _triple  将输入的int数字，转换为三维的元组 eg：1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # 将参数分解为空间和时间组件，方法是使用不会被卷积的轴上的默认值来掩盖这些值。
        # 可以避免比如两次添加填充之类的无意行为是必要的
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]

        # 利用论文3.5节中公式计算中间通道的数量(M)
        intermed_channels = int(math.floor(
            (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        #  定义空间卷积
        # 由于spatial_kernel_size, batch_norm 和 RELU,
        # 空间卷积实际上是一个2D的卷积
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)  # 对数据进行归一化处理
        self.relu = nn.ReLU()  ##   nn.Tanh()   or   nn.ReLU(inplace=True)

        # 时间卷积实际上是一个一维的卷积，但是在模型构造函数中添加了批处理规范和ReLU，而不是在这里。
        # 这是一种有意的设计选择，目的是让此模块在外部表现与标准Conv3D相同，以便在任何其他代码库中轻松重用
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    #  前向传播
    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x


# ------------------------2、定义时空卷积 ST_Block（3DCNN）-------------------

# ------------------------3、定义视频增强模型-------------------
class STVEN_Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=4):
        super(STVEN_Generator, self).__init__()

        # 下采样层：输入n_input_channel,经过Conv_1,输出64通道,最终输出结果为：64*T*128*128
        layers = []
        n_input_channel = 3 + c_dim  # image + label channels
        layers.append(nn.Conv3d(n_input_channel, conv_dim, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3),
                                bias=False))
        layers.append(
            nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))  # affine：当设为true，给该层添加可学习的仿射变换参数。
        # track_running_stats：当设为true，记录训练过程中的均值和方差
        layers.append(nn.ReLU(inplace=True))

        # 输入64通道，输出128通道，经过Conv_2,最终输出结果为：128*T*64*64
        curr_dim = conv_dim

        layers.append(
            nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1), bias=False))
        layers.append(nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # 输入128通道，经过Conv_3,最终输出结果为：512*（T/2）*32*32
        curr_dim = curr_dim * 2

        layers.append(
            nn.Conv3d(curr_dim, curr_dim * 4, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), bias=False))
        layers.append(nn.InstanceNorm3d(curr_dim * 4, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 4

        # 调用ST_Block：输入512通道，输出512通道，重复6次，输出512*（T/2）*32*32。
        for i in range(repeat_num):
            layers.append(SpatioTemporalConv(curr_dim, curr_dim, [3, 3, 3], stride=(1, 1, 1), padding=[1, 1, 1]))

        # 上采样层：输入512通道，输出128通道，经过DConv_1,最终输出结果为：128*T*64*64
        layers2 = []
        layers2.append(
            nn.ConvTranspose3d(curr_dim, curr_dim // 4, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1),
                               bias=False))
        layers2.append(nn.InstanceNorm3d(curr_dim // 4, affine=True, track_running_stats=True))
        layers2.append(nn.ReLU(inplace=True))

        # 输入128通道，经过DConv_2,最终输出结果为：64*T*128*128
        curr_dim = curr_dim // 4

        layers3 = []
        layers3.append(
            nn.ConvTranspose3d(curr_dim, curr_dim // 2, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1),
                               bias=False))
        layers3.append(nn.InstanceNorm3d(curr_dim // 2, affine=True, track_running_stats=True))
        layers3.append(nn.ReLU(inplace=True))

        # 输入64通道，经过DConv_3,最终输出结果为：3*T*128*128
        curr_dim = curr_dim // 2

        layers4 = []
        layers4.append(nn.Conv3d(curr_dim, 3, kernel_size=(1, 7, 7), stride=(1, 1, 1), padding=(0, 3, 3), bias=False))
        layers4.append(nn.Tanh())

        self.down3Dmain = nn.Sequential(*layers)

        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)
        self.layers4 = nn.Sequential(*layers4)

    # 前向传播
    def forward(self, x, c):
        # 在空间上复制和连接域信息。
        c = c.view(c.size(0), c.size(1), 1, 1, 1)  # 展开时对行列数进行指定
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3), x.size(4))

        x0 = torch.cat([x, c], dim=1)  # torch.cat(seq,dim,out=None)
        # 其中seq表示要连接的两个序列，以元组的形式给出，例如:seq=(a,b), a,b 为两个可以连接的序列
        # dim 表示以哪个维度连接，dim=0, 横向连接，dim=1,纵向连接

        x0 = self.down3Dmain(x0)

        x1 = self.layers2(x0)
        x2 = self.layers3(x1)
        x3 = self.layers4(x2)

        out = x3 + x  # 最终输出为残差连接

        return out

