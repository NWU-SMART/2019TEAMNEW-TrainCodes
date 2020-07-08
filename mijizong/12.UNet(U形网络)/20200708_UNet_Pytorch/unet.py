import torch
from torch import nn


# 定义卷积层代码块，进行两次卷积，方便调用
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # 第一层
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            # 第二层
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# 模型实现
class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)  # in_ch*512*512 -> 64*508*508
        self.pool1 = nn.MaxPool2d(2)        # 64*508*508 -> 64*254*254
        self.conv2 = DoubleConv(64, 128)    # 64*254*254 -> 128*250*250
        self.pool2 = nn.MaxPool2d(2)        # 128*250*250 -> 128*125*125
        self.conv3 = DoubleConv(128, 256)   # 128*125*125 -> 256*121*121
        self.pool3 = nn.MaxPool2d(2)        # 256*121*121 -> 256*60*60
        self.conv4 = DoubleConv(256, 512)   # 256*60*60 -> 512*56*56
        self.pool4 = nn.MaxPool2d(2)        # 512*56*56 -> 512*28*28
        self.conv5 = DoubleConv(512, 1024)  # 512*28*28 -> 1024*24*24

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)   # 1024*24*24 -> 512*48*48
        self.conv6 = DoubleConv(1024, 512)                      # 1024*48*48 -> 512*44*44
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)    # 512*44*44 -> 256*88*88
        self.conv7 = DoubleConv(512, 256)                       # 512*88*88 -> 256*84*84
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)    # 256*84*84 -> 128*168*168
        self.conv8 = DoubleConv(256, 128)                       # 256*168*168 -> 128*164*164
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)     # 128*164*164 -> 64*328*328
        self.conv9 = DoubleConv(128, 64)                        # 128*328*328 -> 64*324*324
        self.conv10 = nn.Conv2d(64, out_ch, 1)                  # 64*324*324 -> out_ch*324*324

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        return c10
