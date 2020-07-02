# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/7/2
#  @Author: MiJizong
#  @Content: ResNet——Pytorch
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: Null
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相关的包
# 2、数据预处理
# 3、模型定义
#      定义ResNet的残差模块residual_module
#      定义ResNet的网络构建模块
# 4、模型调用与可视化
# 第三步也可以单独写程序调用
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchviz import make_dot
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1个GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在


#  -------------------------- 1、导入需要包 -------------------------------

# ------------------------ 2、数据预处理 -----------------------------------
# 图像预处理
transform = transforms.Compose([                     # 把多个步骤整合到一起
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),               # 图像一半概率翻转，一半不翻转
    transforms.RandomCrop(32),                       # 图像随机裁剪为32x32
    transforms.ToTensor()])                          # 将PILImage转变为torch.FloatTensor的数据形式

# 加载本地数据集
train_dataset = dsets.CIFAR10(root='./data/',
                              train=True,
                              transform=transform,
                              download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

# ------------------------ 2、数据预处理 -----------------------------------

# resnet做加和操作，因此用add函数，
# googlenet以及densenet做filter的拼接，因此用concatenate
# add和concatenate的区别参考链接：https://blog.csdn.net/u012193416/article/details/79479935

#  -------------------------- 3、模型定义 ---------------------------------

# 先定义3×3的卷积，方便后面直接调用
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,
                     stride=stride,padding=1,bias=False)
# 定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(ResidualBlock,self).__init__()

        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels,out_channels)  # 卷积操作
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  # 层叠加  每组ResidualBlock的第一个 见pdf左分支
        out = self.relu(out)
        return out


# 定义整个ResNet
class ResNet(nn.Module):

    def __init__(self,block,layers,num_class=10):
        super(ResNet,self).__init__()
        self.in_channels = 16      # 输入的通道数

        self.conv = conv3x3(3,16)  # 3x3卷积
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block,16,layers[0])
        self.layer2 = self.make_layer(block,32,layers[1],stride=2)
        self.layer3 = self.make_layer(block,64,layers[2],stride=2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64,num_class)  # 全连接层

    def make_layer(self,block,out_channels,blocks,stride=1):
        downsample = None

        # 统一block的通道数   见pdf左分支
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels,out_channels,stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        # pdf中Add层
        layers = []
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        self.in_channels = out_channels

        # 添加新层
        for i in range(1,blocks):
            layers.append(block(out_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0),-1)  # 平展
        out = self.fc(out)
        return out

resnet = ResNet(ResidualBlock,[2,2,2])  # 3层，每个层有两个ResidualBlock堆叠
print(resnet)

#  -------------------------- 3、模型定义 ---------------------------------

# -----------------------------4、模型调用与可视化--------------------------
resnet = resnet.cuda()


criterion = nn.CrossEntropyLoss()                         # 损失函数
lr = 0.001                                                # 学习率
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)  # 优化器

EPOCH = 5
loss_list = []  # 建立一个loss的列表，以保存每一次loss的数值

for epoch in range(EPOCH):
    for i, (images, labels) in enumerate(train_loader):     # 加载数据
        images, labels = images.cuda(), labels.cuda()
        images = Variable(images)                           # x变为variable类型
        labels = Variable(labels)                           # y变为variable类型
        outputs = resnet(images)                            # 输出
        loss = criterion(outputs, labels)                   # 损失函数
        if i % 20 == 0:
            loss_list.append(loss)                          # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()                               # 梯度清零
        loss.backward()                                     # 反向传播
        optimizer.step()                                    # 更新
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, EPOCH, loss.item()))  # 打印loss

vis_graph = make_dot(outputs, params=dict(list(resnet.named_parameters()) + [('x', images)]))
vis_graph.view()

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
# -----------------------------4、模型调用与可视化-----------------------------
