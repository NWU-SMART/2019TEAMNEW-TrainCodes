# ----------------开发者信息-------------------------------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月10日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息-------------------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、定义网络模型 -------------------------------
# 3x3卷积,输入前后的特征图大小不变
def conv3x3(_in, _out):
    return nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=1)

# 3x3卷积 + ReLU
class ConvRelu(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.conv = conv3x3(_in, _out)  # 卷积
        self.activation = nn.ReLU(inplace=True)  # ReLU
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

# 定义解码层
class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockV2, self).__init__()

        self.in_channels = in_channels  # 输入通道
        self.middle_channels = middle_channels  # 中间通道
        self.out_channels = out_channels  # 输出通道

        self.up = F.interpolate  # 上采样
        self.cr1 = ConvRelu(in_channels, middle_channels)   # 调用ConvRelu
        self.cr2 = ConvRelu(middle_channels, out_channels)  # 调用ConvRelu

    def forward(self, x):
        x = self.up(x, scale_factor=2, mode='bilinear', align_corners=False)  # 上采样，扩大两倍，双线性插值
        x = self.cr2(self.cr1(x))
        return x

# 使用 resnet 的 U-net 网络
class AttentionResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_filters=32, encoder_depth=34, pretrained=True):
        super(AttentionResNet, self).__init__()
        self.in_channels = in_channels  # 输入通道，3
        self.out_channels = out_channels  # 输出通道 1
        self.num_filters = num_filters  # filter 32

        # 设置的数值不同，调用不同的 Resnet,分别为Resnet34, Resnet101, Resnet152
        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')


        self.pool = nn.MaxPool2d(2, 2) # 最大池化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活

        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        # 第一层：Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 第二层：BatchNorm2d(64)
        # 第三层: ReLU(inplace=True)
        # 第四层：MaxPool2d((2,2)), 下采样 ，根据论文输入大小为128x128，此处大小为64x64
        # 输入通道3，输出通道64

        self.conv2 = self.encoder.layer1
        # 三个BasicBlock,输入通道64，输出通道64 ，未进行下采样，大小为64x64
        self.conv3 = self.encoder.layer2
        # 四个BasicBlock，输入通道64，输出通道128 ，进行下采样，大小为32x32
        self.conv4 = self.encoder.layer3
        # 六个BasicBlock，输入通道128，输出通道256 ，进行下采样，大小为16x16
        self.conv5 = self.encoder.layer4
        # 三个BasicBlock，输入通道256，输出通道512 ，进行下采样，大小为8x8

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 16, num_filters * 8)  # 上采样
        # 第一层：Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
        # 第二层：Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU

        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 16, num_filters * 8)  # 上采样
        # 第一层：Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
        # 第二层：Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU

        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 16, num_filters * 8)  # 上采样
        # 第一层：Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
        # 第二层：Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU

        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 8, num_filters * 2)  # 上采样
        # 第一层：Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
        # 第二层：Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 4, num_filters * 4)  # 上采样
        # 第一层：Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
        # 第二层：Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU

        self.dec1 = DecoderBlockV2(num_filters * 4, num_filters * 4, num_filters)  # 上采样
        # 第一层：Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
        # 第二层：Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU


        self.attention_map = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            # Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) + ReLU
            nn.Conv2d(num_filters, out_channels, kernel_size=1)
            # Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1))
        )
    # 前向传播
    def forward(self, x):
        conv1 = self.conv1(x)      # 3-->64      大小：64x64
        conv2 = self.conv2(conv1)  # 64-->64     大小：64x64
        conv3 = self.conv3(conv2)  # 64-->128    大小：32x32
        conv4 = self.conv4(conv3)  # 128-->256   大小：16x16
        conv5 = self.conv5(conv4)  # 256-->512   大小：8x8

        pool = self.pool(conv5)   # 下采样， 大小:4x4

        center = self.center(pool)  # 上采样，大小8x8

        dec5 = self.dec5(torch.cat([center, conv5], 1))   # 8x8的两组图片通道拼接后，继续上采样变为16x16
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))     # 16x16的两组图片通道拼接后，继续上采样变为32x32
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))     # 32x32的两组图片通道拼接后，继续上采样变为64x64
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))     # 64x64的两组图片通道拼接后，继续上采样变为128x128
        dec1 = self.dec1(dec2)

        x = self.attention_map(dec1) #此处生成特征图 Gatt ，特征图大小为，128x128x1
        return x
print(AttentionResNet())
#  -------------------------- 2、定义网络模型 -------------------------------
