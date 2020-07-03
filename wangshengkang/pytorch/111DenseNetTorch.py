# -*- coding: utf-8 -*-
# @Time: 2020/6/28 7:53
# @Author: wangshengkang
# ------------------------------------代码布局-----------------------------------------
# 1导入相关包
# 2设置参数
# 3数据预处理
# 4构建densenet模型
# 5调用模型，进行训练
#6测试
# ------------------------------------代码布局-----------------------------------------
# ------------------------------------1导入相关包-----------------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
import argparse
import os
# ------------------------------------1导入相关包-----------------------------------------
# ------------------------------------2设置参数-----------------------------------------
# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='denoiseAE')
# 调用add_argument()方法添加参数
parser.add_argument('--path', default='./', type=str, help='the path to dataset')
parser.add_argument('--batchsize', default='200', type=int, help='batchsize')
parser.add_argument('--gpu', default='5', type=str, help='choose which gpu to use')
parser.add_argument('--epochs', default='5', type=int, help='the number of epochs')
# 使用parse_args()解析添加的参数
opt = parser.parse_args()
# ------------------------------------2设置参数-----------------------------------------
# ------------------------------------3数据预处理-----------------------------------------
path = opt.path
paths = [
    path + 'cifar-10-batches-py/data_batch_1',
    path + 'cifar-10-batches-py/data_batch_2',
    path + 'cifar-10-batches-py/data_batch_3',
    path + 'cifar-10-batches-py/data_batch_4',
    path + 'cifar-10-batches-py/data_batch_5'
]

# python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存
# 到文件中去，永久存储；通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。
# 从file中读取一个字符串，并将它重构为原来的python对象。file:类文件对象，有read()和readline()接口。
data_batch_1 = pickle.load(open(paths[0], 'rb'), encoding='bytes')
data_batch_2 = pickle.load(open(paths[1], 'rb'), encoding='bytes')
data_batch_3 = pickle.load(open(paths[2], 'rb'), encoding='bytes')
data_batch_4 = pickle.load(open(paths[3], 'rb'), encoding='bytes')
data_batch_5 = pickle.load(open(paths[4], 'rb'), encoding='bytes')

train_X_1 = data_batch_1[b'data']
train_X_1 = train_X_1.reshape(10000, 3, 32, 32).astype('float')
train_Y_1 = data_batch_1[b'labels']

train_X_2 = data_batch_2[b'data']
train_X_2 = train_X_2.reshape(10000, 3, 32, 32).astype('float')
train_Y_2 = data_batch_2[b'labels']

train_X_3 = data_batch_3[b'data']
train_X_3 = train_X_3.reshape(10000, 3, 32, 32).astype('float')
train_Y_3 = data_batch_3[b'labels']

train_X_4 = data_batch_4[b'data']
train_X_4 = train_X_4.reshape(10000, 3, 32, 32).astype('float')
train_Y_4 = data_batch_4[b'labels']

train_X_5 = data_batch_5[b'data']
train_X_5 = train_X_5.reshape(10000, 3, 32, 32).astype('float')
train_Y_5 = data_batch_5[b'labels']

train_X = np.row_stack((train_X_1, train_X_2))  # 行合并
train_X = np.row_stack((train_X, train_X_3))
train_X = np.row_stack((train_X, train_X_4))
train_X = np.row_stack((train_X, train_X_5))

train_Y = np.row_stack((train_Y_1, train_Y_2))
train_Y = np.row_stack((train_Y, train_Y_3))
train_Y = np.row_stack((train_Y, train_Y_4))
train_Y = np.row_stack((train_Y, train_Y_5))
train_Y = train_Y.astype('int32')
train_Y = train_Y.reshape(50000)  # 将2d变为1d

test_batch = pickle.load(open('cifar-10-batches-py/test_batch', 'rb'), encoding='bytes')
test_X = test_batch[b'data']
test_X = test_X.reshape(10000, 3, 32, 32).astype('float')
test_Y = test_batch[b'labels']
test_Y = np.array(test_Y)

train_X /= 255
test_X /= 255
train_X = torch.from_numpy(train_X).float()  # 变为tensor
train_Y = torch.from_numpy(train_Y).long()
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

trainset = TensorDataset(train_X, train_Y)
testset = TensorDataset(test_X, test_Y)
trainloader = DataLoader(trainset, batch_size=opt.batchsize)
testloader = DataLoader(testset, batch_size=opt.batchsize)
# ------------------------------------3数据预处理-----------------------------------------
# ------------------------------------4构建densenet模型-----------------------------------------
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    '''
    block_config=(12, 12, 12),表示三个denseblock里面分别有12,12,12个denselayer
    模型结构 (conv norm relu) (denseblock transition denseblock transition denseblock) 
    norm (relu avgpooling) classifier
    
    '''

    def __init__(self, growth_rate=12, block_config=(12, 12, 12),
                 num_init_features=24, bn_size=4, drop_rate=0.2, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # block_config里面有3个值，所以有3个denseblock
        for i, num_layers in enumerate(block_config):
            # 创建denseblock
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            # 将创建的denseblock加到模型里
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # 只在前两个denseblock后面加上transition，最后一个不加
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 池化，然后将3维变为1维
        out = F.avg_pool2d(out, kernel_size=8, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
# ------------------------------------4构建densenet模型-----------------------------------------
# ------------------------------------5调用模型，进行训练-----------------------------------------
net = DenseNet()
# print(net)
# from torchsummary import summary
# summary(net,(3,32,32))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
# 指定设备
device = torch.device("cuda:%s" % opt.gpu if torch.cuda.is_available() else "cpu")
# 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
net.to(device)

for epoch in range(opt.epochs):

    running_loss = 0.

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print('[%d, %5d] loss: %.4f' % (epoch + 1, (i + 1) * opt.batchsize, loss.item()))

print('Finished Training')
# torch.save(net, 'DenseNet.pkl')
# net = torch.load('DenseNet.pkl')

# ------------------------------------5调用模型，进行训练-----------------------------------------
# ------------------------------------6测试-----------------------------------------
correct = 0
total = 0
with torch.no_grad():
    for i,data in enumerate(testloader):
    #for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)  # 求每一行的最大值，以及最大值的列标，列标作为标签
        total += labels.size(0)#每个batch的数量都加进来
        correct += (predicted == labels).sum().item()#将正确的数量加进来。sum()相加，item()将tensor变为int

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))#乘100，将小数变为百分号形式

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(opt.batchsize):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# 用于算每个类别的准确率
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
# ------------------------------------6测试-----------------------------------------