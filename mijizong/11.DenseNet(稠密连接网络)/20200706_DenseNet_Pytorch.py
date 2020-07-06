# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/7/6
#  @Author: MiJizong
#  @Content: DenseNet——Pytorch
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: Null
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相关的包
# 2、读取数据
# 3、建立稠密连接网络模型
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import LabelBinarizer
from torch.autograd import Variable
from torchviz import make_dot
import matplotlib.pyplot as plt
"""GPU设置为按需增长"""
import os
import tensorflow.compat.v1 as tf   # 使用1.0版本的方法
tf.disable_v2_behavior()            # 禁用2.0版本的方法
# 指定第一块GPU可用
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、读取数据 -------------------------------
data_batch_1 = pickle.load(open("./data/cifar-10-batches-py/data_batch_1", 'rb'), encoding='bytes')
data_batch_2 = pickle.load(open("./data/cifar-10-batches-py/data_batch_2", 'rb'), encoding='bytes')
data_batch_3 = pickle.load(open("./data/cifar-10-batches-py/data_batch_3", 'rb'), encoding='bytes')
data_batch_4 = pickle.load(open("./data/cifar-10-batches-py/data_batch_4", 'rb'), encoding='bytes')
data_batch_5 = pickle.load(open("./data/cifar-10-batches-py/data_batch_5", 'rb'), encoding='bytes')

train_X_1 = data_batch_1[b'data']
train_X_1 = train_X_1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_1 = data_batch_1[b'labels']

train_X_2 = data_batch_2[b'data']
train_X_2 = train_X_2.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_2 = data_batch_2[b'labels']

train_X_3 = data_batch_3[b'data']
train_X_3 = train_X_3.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_3 = data_batch_3[b'labels']

train_X_4 = data_batch_4[b'data']
train_X_4 = train_X_4.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_4 = data_batch_4[b'labels']

train_X_5 = data_batch_5[b'data']
train_X_5 = train_X_5.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_5 = data_batch_5[b'labels']

train_X = np.row_stack((train_X_1, train_X_2))  # 行合并 data
train_X = np.row_stack((train_X, train_X_3))
train_X = np.row_stack((train_X, train_X_4))
train_X = np.row_stack((train_X, train_X_5))

train_Y = np.row_stack((train_Y_1, train_Y_2))  # 行合并 labels
train_Y = np.row_stack((train_Y, train_Y_3))
train_Y = np.row_stack((train_Y, train_Y_4))
train_Y = np.row_stack((train_Y, train_Y_5))
train_Y = train_Y.reshape(50000, 1).transpose(0, 1).astype("int32")  # transpose()坐标互换
train_Y = LabelBinarizer().fit_transform(train_Y)  # 对y进行one-hot编码

test_batch = pickle.load(open("./data/cifar-10-batches-py/test_batch", 'rb'), encoding='bytes')
test_X = test_batch[b'data']
test_X = test_X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
test_Y = test_batch[b'labels']
test_Y = LabelBinarizer().fit_transform(test_Y)

train_X /= 255
test_X /= 255

train_X = Variable(torch.from_numpy(train_X)).float()  # 变为variable数据类型
train_Y = Variable(torch.from_numpy(train_Y)).float()
test_X = Variable(torch.from_numpy(test_X)).float()
test_Y = Variable(torch.from_numpy(test_Y)).float()

train_X = train_X.permute(0, 3, 2, 1)
test_X = test_X.permute(0, 3, 2, 1)
#  -------------------------- 2、读取数据 ------------------------------------------

#  -------------------------- 3、建立稠密连接网络模型 -------------------------------
# 稠密层函数
class Bottleneck(nn.Module):
    def __init__(self,in_channel,growth_rate):
        super(Bottleneck,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv1 = nn.Conv2d(in_channel,4*growth_rate,kernel_size=1,bias=False)  # 第一层卷积，卷积核1x1
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate,growth_rate,kernel_size=3,padding=1,bias=False)  # 第二层卷积，卷积核3x3
        self.drop = nn.Dropout(0.2)

    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))  # BN → rule → conv1
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)  # 将x与out按列拼接，形成新的out
        return self.drop(out)

# 传输层(过渡层)函数
class Transition(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Transition,self).__init__()
        self.bn = nn.BatchNorm2d(in_channel)
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,bias=False)

    def forward(self,x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out,2)
        return out



# 定义DenseNet层
class DenseNet(nn.Module):
    def __init__(self,block,nblocks,growth_rate=12,reduction=0.5,num_classer=10):
        super(DenseNet,self).__init__()
        self.growth_rate = growth_rate  # 定义增长率
        num_channel = 2*growth_rate  # 初始化输出通道 2*12
        self.conv1 = nn.Conv2d(3,num_channel,kernel_size=3,stride=1,padding=1,bias=False)  # 输入3 输出24

        # --- 3个Dense层 -----
        self.dense1 = self.DenseBlock(block,num_channel,nblocks[0])  # 定义2个Bottleneck 输入通道24
        num_channel += nblocks[0]*growth_rate  # 初始输入通道数为24，经过第一个Bottleneck后变为12 相加后输入第二个Bottleneck，再相加变为48
        out_channel = int(math.floor(num_channel*reduction))  # 压缩通道数
        self.trans1 = Transition(num_channel,out_channel)   # 48-> 24
        num_channel = out_channel   # 24

        self.dense2 = self.DenseBlock(block,num_channel,nblocks[1])  # 定义5个Bottleneck 输入通道24
        num_channel += nblocks[1]*growth_rate   # 24 + 12*5 = 84
        out_channel = int(math.floor(num_channel*reduction))  # 压缩通道数
        self.trans2 = Transition(num_channel,out_channel)  # 84 -> 42
        num_channel = out_channel   # 42

        self.dense3 = self.DenseBlock(block,num_channel,nblocks[2])  # 定义4个Bottleneck 输入通道42
        num_channel += nblocks[2]*growth_rate  # 42+12*4 =90
        out_channel = int(math.floor(num_channel*reduction))  # 压缩通道数
        self.trans3 = Transition(num_channel,out_channel)  # 90 -> 45
        num_channel = out_channel  # 45

        self.dense4 = self.DenseBlock(block,num_channel,nblocks[3])  # 定义6个Bottleneck 输入通道45
        num_channel += nblocks[3]*growth_rate   # 45+12*6 = 117
        # --- 4个Dense层 -----

        self.bn = nn.BatchNorm2d(num_channel)
        self.linear = nn.Linear(num_channel,num_classer)  # 117 -> 10

    # 稠密块函数
    def DenseBlock(self, block,in_channel,nblock):
        layers = []
        # 构建nblock个稠密层
        for ii in range(nblock):
            layers.append(block(in_channel,self.growth_rate))
            in_channel += self.growth_rate  # channel = channel + 增长率

        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)),4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out

# 定义四个DenseBlock的卷积
def densenet():
    return DenseNet(Bottleneck,[2,5,4,6])

print(densenet())

#  -------------------------- 3、建立稠密连接网络模型 -------------------------------

# 构建模型
model = densenet()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)
loss_fn = nn.BCELoss()
loss_list = []  # 建立一个loss的列表，以保存每一次loss的数值

# --- 计算Loss
for epoch in range(5):
    output = model(train_X)  # 输入训练数据，获取输出
    loss = loss_fn(train_Y, train_X)  # 输出和训练数据计算损失函数
    loss_list.append(loss)
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 5, loss.item()))  # 每训练1个epoch，打印一次损失函数的值
# --- 计算Loss

# --- 输出学习结果
pred_Y = model.predict(test_X)
score = model.evaluate(test_X, test_Y, verbose=0)
print('Test loss =', score[0])
print('Test accuracy =', score[1])

vis_graph = make_dot(output, params=dict(list(model.named_parameters()) + [('x', train_X)]))
vis_graph.view()

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
#  -------------------------- 3、建立稠密连接网络模型 -------------------------------