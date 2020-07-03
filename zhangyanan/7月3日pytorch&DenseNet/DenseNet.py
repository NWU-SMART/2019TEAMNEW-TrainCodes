# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年7月3日
# 修改日期：
# 修改人：
# 修改内容：

# -------------------------------------------------代码布局------------------------------------------------------------#
# 1、读取数据/数据预处理
# 2、建立密集连接卷积网络模型（定义稠密层函数DenseLayer，定义稠密块函数DenseBlock，定义传输层函数TransitionLayer）
# 3、保存模型与模型可视化
# 4、训练


#  -------------------------- 1、导入需要包 -------------------------------
import pickle
import numpy as np
import torch
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
import torch.nn as nn
import torch.nn.functional as F
import torch
#  -------------------------- 2、读取数据 -------------------------------
#   pickle.load 将pickle数据转换为python的数据结构
data_batch_1 = pickle.load(open("cifar-10-batches-py\\data_batch_1", 'rb'),encoding='bytes')
data_batch_2 = pickle.load(open("cifar-10-batches-py\\data_batch_2", 'rb'),encoding='bytes')
data_batch_3 = pickle.load(open("cifar-10-batches-py\\data_batch_3", 'rb'),encoding='bytes')
data_batch_4 = pickle.load(open("cifar-10-batches-py\\data_batch_4", 'rb'),encoding='bytes')
data_batch_5 = pickle.load(open("cifar-10-batches-py\\data_batch_5", 'rb'),encoding='bytes')


train_x_1 = data_batch_1[b'data']
train_x_1 = train_x_1.reshape(10000,3,32,32).astype("float")  # 不用变换通道位置
train_y_1 = data_batch_1[b'labels']

train_x_2 = data_batch_2[b'data']
train_x_2 = train_x_2.reshape(10000,3,32,32).astype("float")
train_y_2 = data_batch_2[b'labels']

train_x_3 = data_batch_3[b'data']
train_x_3 = train_x_3.reshape(10000,3,32,32).astype("float")
train_y_3 = data_batch_3[b'labels']

train_x_4 = data_batch_4[b'data']
train_x_4 = train_x_4.reshape(10000,3,32,32).astype("float")
train_y_4 = data_batch_4[b'labels']

train_x_5 = data_batch_1[b'data']
train_x_5 = train_x_5.reshape(10000,3,32,32).astype("float")
train_y_5 = data_batch_5[b'labels']


# 矩阵的合并 行合并：np.row_stack()  列合并：np.column_stack()
train_x = np.row_stack((train_x_1,train_x_2))
train_x = np.row_stack((train_x,train_x_3))
train_x = np.row_stack((train_x,train_x_4))
train_x = np.row_stack((train_x,train_x_5))

train_y = np.row_stack((train_y_1,train_y_2))
train_y = np.row_stack((train_y,train_y_3))
train_y = np.row_stack((train_y,train_y_4))
train_y = np.row_stack((train_y,train_y_5))

train_y = train_y.reshape(50000,1).transpose(0,1).astype("int32")  # .reshape 5000行1列 ，transport（1，0）表示行与列调换了位置


encoder = LabelBinarizer()
train_y = encoder .fit_transform(train_y)     # 类别标签转为onehot编码

test_batch = pickle.load(open("test_batch", 'rb'),encoding='bytes')
test_x = test_batch[b'data']
test_x = test_x.reshape(10000,3,32,32).astype("float")
test_y = test_batch[b'labels']
test_y = encoder.fit_transform(test_y)        # onehot编码

train_x /= 255    # 归一化
test_x /= 255     # 归一化



train_x = Variable(torch.from_numpy(train_x)).float()
test_x = Variable(torch.from_numpy(test_x)).float()
train_y = Variable(torch.from_numpy(train_y)).float()
test_y = Variable(torch.from_numpy(test_y)).float()


#  -------------------------- 3、建立稠密连接网络模型 -------------------------------

# 稠密层函数
class Bottleneck(nn.Module):
    def __init__(self,in_planes,growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)                                                    # 正则
        self.conv1 = nn.Conv2d(in_planes,4*growth_rate,kernel_size=1,bias=False)                # 卷积
        self.bn2 = nn.BatchNorm2d(4*growth_rate)                                                # 正则
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3,padding=1,bias=False)  # 卷积

    def forward(self,x):
        out = self.conv1(F.relu(self.bn1(x)))    # 正则，relu激活,卷积
        out = self.conv2(F.relu(self.bn2(out)))  # 正则，relu激活,卷积

        out = torch.cat([out,x],1)               # 将输入和输出按维数1拼接（横着拼)

        return out

# 传输层函数（稠密块和块之间的连接）
class Transition(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(Transition,self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes,out_planes,kernel_size=1,bias=False)

    def forward(self,x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out,2)              # 使用平均池化来改变特征图的大小

        return out

import math

class DenseNet(nn.Module):
    def __init__(self,block,nblocks,growth_rate=12,reduction=0.5,num_classes=10):
        super(DenseNet, self).__init__()
        '''
           block: bottleneck
           nblock: a list, the elements is number of bottleneck in each denseblock
           growth_rate: channel size of bottleneck's output
           reduction: 
        '''
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3,num_planes,kernel_size=3,padding=1,bias=False)

        # 第一个稠密块
        self.dense1 = self._make_dense_layers(block,num_planes,nblocks[0])      # 第一个稠密层
        num_planes += nblocks[0] * growth_rate                                  # 改变输入通道数
        out_planes = int(math.floor(num_planes * reduction))                    # 通过乘以压缩率来减少输出通道数
        self.trans1 = Transition(num_planes,out_planes)                         # 调用传输层
        num_planes = out_planes

        # 第二个稠密块
        self.dense2 = self._make_dense_layers(block,num_planes,nblocks[1])      # 第二个稠密层
        num_planes += nblocks[1] * growth_rate                                  # 改变输入通道数
        out_planes = int(math.floor(num_planes * reduction))                    # 通过乘以压缩率来减少输出通道数
        self.trans2 = Transition(num_planes,out_planes)                         # 调用传输层
        num_planes = out_planes

        # 第三个稠密块
        self.dense3 = self._make_dense_layers(block,num_planes,nblocks[2])      # 第三个稠密层
        num_planes += nblocks[2] * growth_rate                                  # 改变输入通道数
        out_planes = int(math.floor(num_planes * reduction))                    # 通过乘以压缩率来减少输出通道数
        self.trans3 = Transition(num_planes,out_planes)                         # 调用传输层
        num_planes = out_planes

        # 第四个稠密块（没有传输层了）
        self.dense4 = self._make_dense_layers(block,num_planes,nblocks[3])      # 第三个稠密层
        num_planes += nblocks[3] * growth_rate                                  # 改变输入通道数

        self.bn = nn.BatchNorm2d(num_planes)                                    # 正则
        self.linear = nn.Linear(num_planes,num_classes)                         # 10分类

    def _make_dense_layers(self,block,in_planes,nblock):
        layers = []
        for i in range(nblock):                               # 将n个稠密层写成Sequential的形式
            layers.append(block(in_planes,self.growth_rate))
            in_planes += self.growth_rate

        return nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)                                   # 先卷积
        out = self.trans1(self.dense1(out))                   # 第一个稠密块
        out = self.trans2(self.dense2(out))                   # 第二个稠密块
        out = self.trans3(self.dense3(out))                   # 第三个稠密块
        out = self.dense4(out)                                # 第四个稠密块
        out = self.F.avg_pool2d(F.relu(self.bn(out)),4)       # 正则 relu 平均池化
        out = out.view(out.size(0),-1)                        # 拉平
        out = self.linear(out)                                # 全连接 10分类

        return out

def densenet():
    return DenseNet(Bottleneck,[2,5,4,6])                    # nblock:四个稠密块，每个稠密块分别有2，5,4,6个稠密层

print(densenet())


model = densenet()

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
Epoch = 5


## 开始训练 ##
for t in range(Epoch):

    # x = model(train_x.to(device))          # 向前传播
    # loss = loss_fn(x, train_y.to(device))  # 计算损失
    x = model(train_x)  # 向前传播
    loss = loss_fn(x, train_y)  # 计算损失

    if (t + 1) % 1 == 0:        # 每训练1个epoch，打印一次损失函数的值
        print(loss.item())

    if (t + 1) % 5 == 0:
        torch.save(model.state_dict(), "./pytorch_densenet_model.pkl")  # 每5个epoch保存一次模型
        print("save model")

    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度







