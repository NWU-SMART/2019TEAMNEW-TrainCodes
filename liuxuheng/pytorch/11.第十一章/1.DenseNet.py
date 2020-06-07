# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月2日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、读取数据 -------------------------------
data_batch_1 = pickle.load(open("cifar-10-batches-py/data_batch_1", 'rb'), encoding='bytes') #读取数据
data_batch_2 = pickle.load(open("cifar-10-batches-py/data_batch_2", 'rb'), encoding='bytes')
data_batch_3 = pickle.load(open("cifar-10-batches-py/data_batch_3", 'rb'), encoding='bytes')
data_batch_4 = pickle.load(open("cifar-10-batches-py/data_batch_4", 'rb'), encoding='bytes')
data_batch_5 = pickle.load(open("cifar-10-batches-py/data_batch_5", 'rb'), encoding='bytes')

train_X_1 = data_batch_1[b'data'] # 读取train_X
train_X_1 = train_X_1.reshape(10000, 3, 32, 32).astype("float") #reshape
train_Y_1 = data_batch_1[b'labels'] # 读取train_Y

train_X_2 = data_batch_2[b'data']
train_X_2 = train_X_2.reshape(10000, 3, 32, 32).astype("float")
train_Y_2 = data_batch_2[b'labels']

train_X_3 = data_batch_3[b'data']
train_X_3 = train_X_3.reshape(10000, 3, 32, 32).astype("float")
train_Y_3 = data_batch_3[b'labels']

train_X_4 = data_batch_4[b'data']
train_X_4 = train_X_4.reshape(10000, 3, 32, 32).astype("float")
train_Y_4 = data_batch_4[b'labels']

train_X_5 = data_batch_5[b'data']
train_X_5 = train_X_5.reshape(10000, 3, 32, 32).astype("float")
train_Y_5 = data_batch_5[b'labels']

train_X = np.row_stack((train_X_1, train_X_2))# 行合并
train_X = np.row_stack((train_X, train_X_3))
train_X = np.row_stack((train_X, train_X_4))
train_X = np.row_stack((train_X, train_X_5))

train_Y = np.row_stack((train_Y_1, train_Y_2))# 行合并
train_Y = np.row_stack((train_Y, train_Y_3))
train_Y = np.row_stack((train_Y, train_Y_4))
train_Y = np.row_stack((train_Y, train_Y_5))

train_X /= 255 # 归一化
train_Y = train_Y.reshape(50000, 1).transpose(0, 1).astype("int32")#reshape
train_Y = LabelBinarizer().fit_transform(train_Y)# 对y进行one-hot编码

train_X = Variable(torch.from_numpy(train_X)).float() #变为variable数据类型
train_Y = Variable(torch.from_numpy(train_Y)).float()

print(train_X.shape)
print(train_Y.shape)

#  -------------------------- 2、读取数据 -------------------------------

#  -------------------------- 3、建立稠密连接网络模型 -------------------------------
# 定义卷积模型
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes) # BN层，in_planes为channel
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False) # 第一层卷积，卷积核1x1
        self.bn2 = nn.BatchNorm2d(4*growth_rate) # BN层
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)#第二层卷积，卷积核3x3
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x))) # 输入x经过BN层后，再过relu激活函数，再过第一层卷积
        out = self.conv2(F.relu(self.bn2(out)))# 输出out经过BN层后，再过relu激活函数，再过第二层卷积
        out = torch.cat([out,x], 1) #将x与out按列拼接，形成新的out
        return out

# 下采样
class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes) # BN层
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False) # 卷积层
    def forward(self, x): 
        out = self.conv(F.relu(self.bn(x)))# 输入x经过BN层，relu层，卷积层
        out = F.avg_pool2d(out, 2) # 输出out经过平均池化,压缩图片大小
        return out 

# 定义DenseNet模型
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate  # 定义增长率12
        num_planes = 2*growth_rate # 初始输出通道定义为24
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False) # 输入3通道，输出24通道

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])#第一个dense层，定义了2个Bottleneck
        num_planes += nblocks[0]*growth_rate # 初始输入通道数为24,经过第一个Bottleneck,变为12，相加为36输入第二个Bottleneck，再次得到12，12+36=48
        out_planes = int(math.floor(num_planes*reduction)) # 压缩通道数
        self.trans1 = Transition(num_planes, out_planes) # 调用下采样函数，48变24
        num_planes = out_planes # 24


        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])#第二个dense层，定义了5个Bottleneck
        num_planes += nblocks[1]*growth_rate #  24-->12 -->36-->12 -->48-->12 -->60-->12 -->72-->12 -->84
        out_planes = int(math.floor(num_planes*reduction))# 压缩通道数
        self.trans2 = Transition(num_planes, out_planes)# 84-->42
        num_planes = out_planes # 42


        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])#第三个dense层，定义了4个Bottleneck
        num_planes += nblocks[2]*growth_rate # 42-->12 -->54-->12 -->66-->12 -->78-->12 -->90
        out_planes = int(math.floor(num_planes*reduction))# 压缩通道数
        self.trans3 = Transition(num_planes, out_planes)# 90-->45
        num_planes = out_planes # 45


        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])#第四个dense层,定义了6个Bottleneck
        num_planes += nblocks[3]*growth_rate # 45-->12-->57-->12-->69-->12-->81-->12-->93-->12-->105-->12-->117
        self.bn = nn.BatchNorm2d(num_planes) # BN层,117
        self.linear = nn.Linear(num_planes, num_classes) #全连接层分类117-->10

    # 稠密块
    def _make_dense_layers(self, block, in_planes, nblock):
        layers = [] # 定义layers
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate #channel = channel + 增长率
        return nn.Sequential(*layers)

    # 正向传播
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 定义四个dense层的卷积层数
def densenet():
    return DenseNet(Bottleneck, [2, 5, 4, 6])
#  -------------------------- 3、建立稠密连接网络模型 -------------------------------

print(densenet())

#  -------------------------- 4、模型训练   --------------------------------
model = densenet()# 定义model
loss_fn = nn.BCELoss() #损失函数
learning_rate = 1e-4 # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #SGD优化器
for epoch in range(5):
    output = model(train_X)# 输入训练数据，获取输出
    loss = loss_fn(train_Y, train_X)# 输出和训练数据计算损失函数
    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播
    optimizer.step()#梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 5, loss.item()))#每训练1个epoch，打印一次损失函数的值
#  -------------------------- 4、模型训练   --------------------------------



