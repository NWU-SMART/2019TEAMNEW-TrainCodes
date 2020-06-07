# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月4日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、定义G网络模型 -------------------------------
# 生成器模型
class Generator(nn.Module):
    def __init__(self,nz,ngf):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入噪声100z,输出 1024x4x4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 输入1024x4x4,输出512x8x8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 输入512x8x8,输出256x16x16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 输入256x16x16,输出128x32x32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 输入128x32x32,输出3x64x64
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # 正向传播
    def forward(self, input):
        output = self.main(input)
        return output

# 查看模型
netG = Generator(100,128)
print(netG)
#  -------------------------- 2、定义G网络模型 -------------------------------

#  -------------------------- 3、定义D网络模型 -------------------------------
# 判别器模型
class Discriminator(nn.Module):
    def __init__(self,ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入3x64x64, 输出64x32x32
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入64x32x32, 输出128x16x16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入128x16x16,输出256x8x8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入256x8x8,输出512x4x4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输入512x4x4,输出1x1x1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # 正向传播
    def forward(self, input):
        output = self.main(input)
        return output

# 查看模型
netD = Discriminator(64)
print(netD)
#  -------------------------- 3、定义D网络模型 -------------------------------



