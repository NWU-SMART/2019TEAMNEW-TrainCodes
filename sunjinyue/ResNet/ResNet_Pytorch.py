# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年7月2日
# 修改日期：
# 修改人：
# 修改内容：
'''
输入参数：
block：残差模块，分为BasicBlock和Bottleneck，resnet18和resnet34中使用是BasicBlock，剩余的使用Bottleneck，BasicBlock中有2个卷积层，Bottleneck中有3个卷积层。
layers：每个block的数目。
resnet18中为layers为[3,4,6,3]，
resnet50中为layers为[3,4,6,3]，
num_classes：输入类别数目
zero_init_residual：对残差模块的最后一层bn层的参数初始化为0，更有利用拟合残差映射
groups：分组卷积，resnext50_32x4d，resnext101_32x8d，wide_resnet50_2，wide_resnet101_2中将会涉及。这里不展开讲分组卷积了。
width_per_group：分组卷积相关。
replace_stride_with_dilation：空洞卷积，可以增加感受野。
norm_layer：归一化层
'''
import torch.nn as nn
import torchvision
import torch
from torchvision.models.resnet import conv1x1, conv3x3


class ResNet(nn.Module):
    # block：残差模块，分为basicblock和bottleneck，resnet18和resnet34中使用是BasicBlock

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        """参数设置部分"""
        # 使用批归一化
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # 第一个残差模块的输入通道数目
        self.inplanes = 64
        # 空洞卷积间隔为1
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        # 分组卷积相关
        self.groups = groups
        self.base_width = width_per_group

        '''模型搭建部分'''
        # 首先是conv7*7 -> bn -> relu -> maxpool
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 残差模块 layer1 -> layer2 -> layer3 -> layer4
        # block为Bottleneck或BasicBlock，根据具体的resnet版本来分，例如layer1=Bottleneck*layer[0]
        # _make_layer中的第二个参数为残差模块的第一个卷积层的输出通道数
        # _make_layer中的第三个参数为该残差模块的重复次数
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # avgpool -> FC，expansion是block最终的输出通道数目与block模块第一个卷积层的输出通道数的比例，BasicBlock中的expansion为1，Bottleneck中的expansion为4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 如果block输入通道数与block的输出通道数目或者尺寸不相同，进行downsample，使得两者通道数以及尺寸相同
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # 后续的block输入的通道数与第一个block的输出通道数相同，不需要设置downsample
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

'''
BasicBlock
resnet18和resnet34中使用是BasicBlock，其主要是由两个 [公式] 卷积层构成，
expansion代表的是残差模块最终输出通道数目和第一个卷积层的输出通道数目的比例，
BasicBlock中的expansion为1。downsample表示如果输入与输出的大小不相同，需要使用卷积核对输入进行通道升维或者降尺寸，
使得输入与输出的通道数相同，这样就能对两者进行直接相加。
'''
class BasicBlock(nn.Module):
    expansion = 1 # 最终输出通道数目和第一个卷积层的输出通道数目的比例

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        # 是否使用批归一化层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # group代表分组卷积，这里不使用分组卷积
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # dilation代表空洞卷积，这里不使用空洞卷积
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # 网络结构
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 对输入进行downsample,是的输入输出大小相同，可以直接相加
        if self.downsample is not None:
            identity = self.downsample(x)

        # 输入与输出相加
        out += identity
        out = self.relu(out)

        return out

'''
除了resnet18和resnet34，剩下的resnet均使用Bottleneck作为残差模块。
Bottleneck主要包括三个卷积层它的expansion为4，
代表残差模块的最终输出通道数为第一个卷积层的输出通道数的4倍。
'''
class Bottleneck(nn.Module):
    expansion = 4  # 最终输出通道数目和第一个卷积层的输出通道数目的比例

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        # 是否使用批归一化层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
                # 第一个卷积层的输出通道数
        width = int(planes * (base_width / 64.)) * groups

        #  网络结构
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 对输入进行downsample, 输入输出大小相同，可以直接相加
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
