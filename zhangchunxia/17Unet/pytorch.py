# -----------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月9日
# 内容:Unet
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 torch的包
# 2、上采样，下采样，模型可视化
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import torch.nn as nn
import torch
from torch import autograd
#  ---------------------- 1、导入需要包 -----------------------------------
# 把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),    # 输入通道，输出通道，卷积核，步长
            nn.BatchNorm2d(out_ch),                 # 正则
            nn.ReLU(inplace=True),                  # 激活
            nn.Conv2d(out_ch, out_ch, 3, padding=1),# 输入通道，输出通道，卷积核，步长
            nn.BatchNorm2d(out_ch),                 # 正则
            nn.ReLU(inplace=True)                   # 激活
        )
    def forward(self,x):
        return self.conv(x)

    # 定义U-Net
class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
            super(UNet, self).__init__()
            # 下采样
            self.conv1 = DoubleConv(in_ch, 64),
            self.pool1 = nn.MaxPool2d(2)  # 图像大小缩小一半

            self.conv2 = DoubleConv(64, 128),
            self.pool2 = nn.MaxPool2d(2)  # 图像大小缩小一半

            self.conv3 = DoubleConv(128, 256),
            self.pool3 = nn.MaxPool2d(2)  # 图像大小缩小一半

            self.conv4 = DoubleConv(256, 512),
            self.pool4 = nn.MaxPool2d(2)  # 图像大小缩小一半

            self.conv5 = DoubleConv(512, 1024),

            # 上采样
            self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.conv6 = DoubleConv(1024, 512)

            self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.conv7 = DoubleConv(512, 256)

            self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.conv8 = DoubleConv(256, 128)

            self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.conv9 = DoubleConv(128, 64)

            self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
            conv1 = self.conv1(x)
            pool1 = self.pool1(conv1)

            conv2 = self.conv2(pool1)
            pool2 = self.pool2(conv2)

            conv3 = self.conv3(pool2)
            pool3 = self.pool3(conv3)

            conv4 = self.conv4(pool3)
            pool4 = self.pool4(conv4)

            conv5 = self.conv5(pool4)

            up6 = self.up6(conv5)
            merge6 = torch.cat([up6, conv4], dim=1)  # 按维度1 （列）拼接，列增加
            conv6 = self.conv6(merge6)

            up7 = self.up7(conv6)
            merge7 = torch.cat([up7, conv3], dim=1)  # 按维度1 （列）拼接，列增加
            conv7 = self.conv7(merge7)

            up8 = self.up8(conv7)
            merge8 = torch.cat([up8, conv2], dim=1)  # 按维度1 （列）拼接，列增加
            conv8 = self.conv8(merge8)

            up9 = self.up9(conv8)
            merge9 = torch.cat([up9, conv1], dim=1)  # 按维度1 （列）拼接，列增加
            conv9 = self.conv9(merge9)

            conv10 = self.conv10(conv9)

            out = nn.Sigmoid()(conv10)

            return out

model = UNet(3, 1)
print(model)