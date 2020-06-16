# ------------------------1、导入需要的模块-------------------------
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import pdb
import torch


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

        # self-definition
        # intermed_channels = int((in_channels+intermed_channels)/2)

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


# ------------------------3、定义基于空间的皮肤注意力模型-------------------
class MixA_Module(nn.Module):
    """ Spatial-Skin attention module"""

    def __init__(self):
        super(MixA_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.AVGpool = nn.AdaptiveAvgPool1d(1)
        self.MAXpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, skin):
        """
            inputs :
                x : input feature maps( B X C X T x W X H)
                skin : skin confidence maps( B X T x W X H)
            returns :
                out : attention value
                spatial attention: W x H
        """
        m_batchsize, C, T, W, H = x.size()
        B_C_TWH = x.view(m_batchsize, C, -1)  # view用于改变输出维度
        B_TWH_C = x.view(m_batchsize, C, -1).permute(0, 2, 1)  # permute函数对第二维第三维进行倒置
        B_TWH_C_AVG = torch.sigmoid(self.AVGpool(B_TWH_C)).view(m_batchsize, T, W, H)  # 平均池化+sigmoid
        B_TWH_C_MAX = torch.sigmoid(self.MAXpool(B_TWH_C)).view(m_batchsize, T, W, H)  # 最大池化+sigmoid
        B_TWH_C_Fusion = B_TWH_C_AVG + B_TWH_C_MAX + skin  # 对应论文3.2节的公式（7）
        Attention_weight = self.softmax(B_TWH_C_Fusion.view(m_batchsize, T, -1))
        Attention_weight = Attention_weight.view(m_batchsize, T, W, H)  # view用于改变输出维度

        # mask1 mul
        output = x.clone()  # 最后的输出（即特征图F）
        for i in range(C):  # 遍历每个通道
            output[:, i, :, :, :] = output[:, i, :, :, :].clone() * Attention_weight  # 对应图片3：输入F与Attention Maps相乘

        return output, Attention_weight


# ------------------------3、定义基于空间的皮肤注意力模型-------------------

# for open-source
# skin segmentation + PhysNet + MixA3232 + MixA1616part4（分区约束）
# ------------------------4、定义rPPGNet--------------------------------
class rPPGNet(nn.Module):
    def __init__(self):
        super(rPPGNet, self).__init__()
        # 输入3通道，输出16通道
        self.ConvSpa1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        # 输入16通道，输出32通道，32*T*64*64
        self.ConvSpa3 = nn.Sequential(
            SpatioTemporalConv(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa4 = nn.Sequential(
            SpatioTemporalConv(32, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa5 = nn.Sequential(
            SpatioTemporalConv(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa6 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa7 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa8 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvSpa9 = nn.Sequential(
            SpatioTemporalConv(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvSpa10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvSpa11 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        # 对应分区约束
        self.ConvPart1 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvPart2 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvPart3 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.ConvPart4 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        # 平均池化
        self.AvgpoolSpa = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.AvgpoolSkin_down = nn.AvgPool2d((2, 2), stride=2)
        self.AvgpoolSpaTem = nn.AvgPool3d((2, 2, 2), stride=2)

        self.ConvSpa = nn.Conv3d(3, 16, [1, 3, 3], stride=1, padding=[0, 1, 1])

        # 对输出进行全局平均池化
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.poolspa = nn.AdaptiveAvgPool3d((64, 1, 1))  # attention to this value

        # skin_branch
        self.skin_main = nn.Sequential(
            nn.Conv3d(32, 16, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.skin_residual = nn.Sequential(
            nn.Conv3d(32, 8, [1, 1, 1], stride=1, padding=0),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )

        self.skin_output = nn.Sequential(
            nn.Conv3d(8, 1, [1, 3, 3], stride=1, padding=[0, 1, 1]),
            nn.Sigmoid(),  ## binary
        )

        self.MixA_Module = MixA_Module()

    def forward(self, x):  # x [3, 64, 128,128]
        x_visual = x

        x = self.ConvSpa1(x)  # x [3, 64, 128,128]--->x [16, 64, 128,128]
        x = self.AvgpoolSpa(x)  # x [16, 64, 128,128]--->x [16, 64, 64,64]

        x = self.ConvSpa3(x)  # x [16, 64, 64,64]--->x [32, 64, 64,64]
        x_visual6464 = self.ConvSpa4(x)  # x [32, 64, 64,64]--->x [32, 64, 64,64]
        x = self.AvgpoolSpa(x_visual6464)  # x [32, 64, 64,64]--->x [32, 64, 32,32]

        ## branch 1: skin segmentation
        x_skin_main = self.skin_main(x_visual6464)  # x [32, 64, 32,32]--->x [8, 64, 64,64]
        x_skin_residual = self.skin_residual(x_visual6464)  # x [32, 64, 32,32]--->x [8, 64, 64,64]
        x_skin = self.skin_output(x_skin_main + x_skin_residual)  # x [8, 64, 64,64]--->x [1, 64, 64,64]
        x_skin = x_skin[:, 0, :, :, :]  # x [64,64,64]   ????

        ## branch 2: rPPG
        x = self.ConvSpa5(x)  # x [32, 64, 32,32]--->x [64, 64, 32,32]
        x_visual3232 = self.ConvSpa6(x)  # x [64, 64, 32,32]--->x [64, 64, 32,32]
        x = self.AvgpoolSpa(x_visual3232)  # x [64, 64, 32,32]--->x [64, 64, 16,16]

        x = self.ConvSpa7(x)  # x [64, 64, 16,16]--->x [64, 64, 16,16]
        x = self.ConvSpa8(x)  # x [64, 64, 16,16]--->x [64, 64, 16,16]
        x_visual1616 = self.ConvSpa9(x)  # x [64, 64, 16,16]--->x [64, 64, 16,16]

        ## SkinA1_loss
        x_skin3232 = self.AvgpoolSkin_down(x_skin)  # x [64,64,64]--->x [64, 32,32]
        # 使用皮肤注意力模块
        x_visual3232_SA1, Attention3232 = self.MixA_Module(x_visual3232, x_skin3232)
        # inputs : x : input feature maps( B X C X T x W X H)   skin : skin confidence maps( B X T x W X H)
        # returns : out : attention value     spatial attention: W x H (32*32)

        x_visual3232_SA1 = self.poolspa(x_visual3232_SA1)  # 全局平均池化  # x [64, 64, 1,1]
        ecg_SA1 = self.ConvSpa10(x_visual3232_SA1).view(-1, 64)  # x [1, 64, 1,1]
        # view(-1,64) -1：不确定变成几行  64：变成64列

        ## SkinA2_loss
        x_skin1616 = self.AvgpoolSkin_down(x_skin3232)  # x [64, 32,32]--->x [64, 16,16]
        x_visual1616_SA2, Attention1616 = self.MixA_Module(x_visual1616, x_skin1616)
        ## Global
        global_F = self.poolspa(x_visual1616_SA2)  # x [64, 64, 1,1]
        ecg_global = self.ConvSpa11(global_F).view(-1, 64)  # x [1, 64, 1,1]

        ## Local
        # 取左下角的8*8块
        Part1 = x_visual1616_SA2[:, :, :, :8, :8]
        Part1 = self.poolspa(Part1)  # x [64, 64, 1,1]
        ecg_part1 = self.ConvPart1(Part1).view(-1, 64)  # x [1, 64, 1,1]
        # 取右下角的8*8块
        Part2 = x_visual1616_SA2[:, :, :, 8:16, :8]
        Part2 = self.poolspa(Part2)  # x [64, 64, 1,1]
        ecg_part2 = self.ConvPart2(Part2).view(-1, 64)  # x [1, 64, 1,1]

        Part3 = x_visual1616_SA2[:, :, :, :8, 8:16]
        Part3 = self.poolspa(Part3)  # x [64, 64, 1,1]
        ecg_part3 = self.ConvPart3(Part3).view(-1, 64)  # x [1, 64, 1,1]

        Part4 = x_visual1616_SA2[:, :, :, 8:16, 8:16]
        Part4 = self.poolspa(Part4)  # x [64, 64, 1,1]
        ecg_part4 = self.ConvPart4(Part4).view(-1, 64)  # x [1, 64, 1,1]

        return x_skin, ecg_SA1, ecg_global, ecg_part1, ecg_part2, ecg_part3, ecg_part4, x_visual6464, x_visual3232



