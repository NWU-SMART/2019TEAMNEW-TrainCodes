# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月3日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch.nn as nn
import torch
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、定义网络模型 -------------------------------
#将常用的卷积封装
class ConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))# 卷积

        conv_relu.append(nn.ReLU())#relu

        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))# 卷积

        conv_relu.append(nn.ReLU())#relu

        self.conv_ReLU = nn.Sequential(*conv_relu)

    # 正向传播
    def forward(self, x):
        out = self.conv_ReLU(x)
        return out

# 定义u-net
class U_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 首先定义左半部分网络
        # 第一层
        self.left_conv_1 = ConvBlock(in_channels=3, middle_channels=64, out_channels=64)# 卷积
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)# 池化
        # 第二层
        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第三层
        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第四层
        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 第五层
        self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)


        # 定义右半部分网络
        # 第五层
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)# 上采样
        self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512) # 卷积
        # 第四层
        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.right_conv_2 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256)
        # 第三层
        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2 ,output_padding=1)
        self.right_conv_3 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128)
        # 第二层
        self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.right_conv_4 = ConvBlock(in_channels=128, middle_channels=64, out_channels=64)
        # 第一层
        self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)

    # 正向传播
    def forward(self, x):

        # 第一层
        feature_1 = self.left_conv_1(x)  # 卷积
        feature_1_pool = self.pool_1(feature_1) # 池化
        # 第二层
        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)
        # 第三层
        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)
        # 第四层
        feature_4 = self.left_conv_4(feature_3_pool)
        feature_4_pool = self.pool_4(feature_4)
        # 第五层
        feature_5 = self.left_conv_5(feature_4_pool)


        # 第五层
        de_feature_1 = self.deconv_1(feature_5) # 上采样
        temp = torch.cat((feature_4, de_feature_1), dim=1) #通道拼接
        de_feature_1_conv = self.right_conv_1(temp)# 卷积
        # 第四层
        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)
        # 第三层
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)
        # 第二层
        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)
        # 第一层
        out = self.right_conv_5(de_feature_4_conv)

        return out #输出
#  -------------------------- 2、定义网络模型 -------------------------------

#  -------------------------- 3、测试网络模型 -------------------------------
# 检验输入与输出shape
if __name__ == "__main__":
    x = torch.rand(size=(8, 3, 224, 224))
    net = U_Net()
    out = net(x)
    print(out.size())
    print("ok")
#  -------------------------- 3、测试网络模型 -------------------------------
