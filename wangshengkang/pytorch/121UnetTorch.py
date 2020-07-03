# -*- coding: utf-8 -*-
# @Time: 2020/7/1 8:32
# @Author: wangshengkang
# @Software: PyCharm

import torch.nn as nn
import torch
from torch import autograd
from torchsummary import summary


# 把常用的2个卷积操作简单封装下
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)  # 64*224*224
        self.pool1 = nn.MaxPool2d(2)  # 64*112*112
        self.conv2 = DoubleConv(64, 128)  # 128*112*112
        self.pool2 = nn.MaxPool2d(2)  # 128*56*56
        self.conv3 = DoubleConv(128, 256)  # 256*56*56
        self.pool3 = nn.MaxPool2d(2)  # 256*28*28
        self.conv4 = DoubleConv(256, 512)  # 512*28*28
        self.pool4 = nn.MaxPool2d(2)  # 512*14*14
        self.conv5 = DoubleConv(512, 1024)  # 1024*14*14
        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 512*28*28
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1 = self.conv1(x)  # 64*224*224
        p1 = self.pool1(c1)  # 64*112*112
        c2 = self.conv2(p1)  # 128*112*112
        p2 = self.pool2(c2)  # 128*56*56
        c3 = self.conv3(p2)  # 256*56*56
        p3 = self.pool3(c3)  # 256*28*28
        c4 = self.conv4(p3)  # 512*28*28
        p4 = self.pool4(c4)  # 512*14*14
        c5 = self.conv5(p4)  # 1024*14*14
        up_6 = self.up6(c5)  # 512*28*28
        merge6 = torch.cat([up_6, c4], dim=1)  # 1024*28*28
        c6 = self.conv6(merge6)  # 512*28*28
        up_7 = self.up7(c6)  # 256*56*56
        merge7 = torch.cat([up_7, c3], dim=1)  # 512*56*56
        c7 = self.conv7(merge7)  # 256*56*56
        up_8 = self.up8(c7)  # 128*112*112
        merge8 = torch.cat([up_8, c2], dim=1)  # 256*112*112
        c8 = self.conv8(merge8)  # 128*112*112
        up_9 = self.up9(c8)  # 64*224*224
        merge9 = torch.cat([up_9, c1], dim=1)  # 128*224*224
        c9 = self.conv9(merge9)  # 64*224*224
        c10 = self.conv10(c9)  # 3*224*224
        # out = nn.Sigmoid()(c10)
        out = self.sigmoid(c10)  # 3*224*224
        return out


net = Unet()
print(net)
summary(net, (3, 224, 224))
