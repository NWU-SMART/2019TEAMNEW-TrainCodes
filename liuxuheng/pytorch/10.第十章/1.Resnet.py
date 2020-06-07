# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月5日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、下载数据及与图像预处理 ---------------------
# 图像预处理
transform = transforms.Compose([
    transforms.Scale(40),
    transforms.RandomHorizontalFlip(),#图像一半概率翻转，一半不翻转
    transforms.RandomCrop(32),# 图像随即裁剪为32x32
    transforms.ToTensor()])# 转为tensor

# 下载CIFAR10数据集
train_dataset = dsets.CIFAR10(root='./data/',
                              train=True,
                              transform=transform,
                              download=False)

# 加载数据集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)
#  --------------------- 2、下载数据及与图像预处理 ---------------------

#  -------------------------- 3、定义模型   --------------------------------
# 定义3x3的卷积，方便之后调用
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# 定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)# 卷积
        self.bn1 = nn.BatchNorm2d(out_channels)# BN层
        self.relu = nn.ReLU(inplace=True)#relu
        self.conv2 = conv3x3(out_channels, out_channels)# 卷积
        self.bn2 = nn.BatchNorm2d(out_channels)# BN层
        self.downsample = downsample # 保证维度统一
    #正向传播
    def forward(self, x):
        residual = x 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)# 保证维度统一
        out += residual # residual+当前feature map
        out = self.relu(out)
        return out

# 定义整个Resnet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16 # channel
        self.conv = conv3x3(3, 16) # 卷积
        self.bn = nn.BatchNorm2d(16) # BN层
        self.relu = nn.ReLU(inplace=True) # relu
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)# 平均池化
        self.fc = nn.Linear(64, num_classes) # 全连接层
    # 构造新层
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        # 当需要特征图需要降维或通道数不匹配的时候调用
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),# 卷积
                nn.BatchNorm2d(out_channels))#  BN层
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))#添加层
        self.in_channels = out_channels

        for i in range(1, blocks):#按照输入个数，添加新层
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    # 正向传播
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)#展平
        out = self.fc(out)
        return out
resnet = ResNet(ResidualBlock, [2, 2, 2]) # 3层，每层两个ResidualBlock堆叠
print(resnet)
#  -------------------------- 3、定义模型   --------------------------------

#  -------------------------- 4、训练模型   --------------------------------
criterion = nn.CrossEntropyLoss()# 损失函数
lr = 0.001# 学习率
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)# 优化器
for epoch in range(5):
    for i, (images, labels) in enumerate(train_loader):# 加载数据
        images = Variable(images)# x变为variable类型
        labels = Variable(labels)# y变为variable类型
        optimizer.zero_grad()# 梯度清零
        outputs = resnet(images)# 输出
        loss = criterion(outputs, labels)# 损失函数
        loss.backward()# 反向传播
        optimizer.step()# 更新
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 5, loss.item()))# 打印loss
#  -------------------------- 4、训练模型   --------------------------------


