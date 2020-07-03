# -*- coding: utf-8 -*-
# @Time: 2020/6/21 16:05
# @Author: wangshengkang
# ------------------------------------代码布局-----------------------------------------
# 1导入相关包
# 2设置参数
# 3构建resnet模型
# 4数据预处理
# 5调用模型，进行训练
# ------------------------------------代码布局-----------------------------------------
# ------------------------------------1导入相关包-----------------------------------------
import torch.nn as nn
import torch.utils.model_zoo as model_zoo  # 导入model_zoo，作用是根据下面的model_urls里的地址加载网络预训练权重。
import gzip
import numpy as np
import os
import cv2
import argparse
import torch
import torchvision
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ------------------------------------1导入相关包-----------------------------------------
# ------------------------------------2设置参数-----------------------------------------
# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='denoiseAE')
# 调用add_argument()方法添加参数
parser.add_argument('--path', default='./', type=str, help='the path to dataset')
parser.add_argument('--batchsize', default='32', type=int, help='batchsize')
parser.add_argument('--gpu', default='7', type=str, help='choose which gpu to use')
parser.add_argument('--epochs', default='5', type=int, help='the number of epochs')
# 使用parse_args()解析添加的参数
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu  # 选择gpu
# ------------------------------------2设置参数-----------------------------------------

# ------------------------------------3构建resnet模型-----------------------------------------

# model_urls这个字典是预训练模型的下载地址。
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# 这里定义了最重要的残差模块，这个是基础版，由两个叠加的3x3卷积组成
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


'''
这里是三个卷积，分别是1x1,3x3,1x1,分别用来压缩维度，卷积处理，恢复维度，
inplane是输入的通道数，plane是输出的通道数，expansion是对输出通道数的倍乘，
plane不再代表输出的通道数，而是block内部压缩后的通道数，输出通道数变为plane*expansion。
'''


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # element-wise add的操作。
        out += residual
        out = self.relu(out)

        return out


'''
resnet共有五个阶段，其中第一阶段为一个7x7的卷积处理，stride为2，然后经过池化
处理，此时特征图的尺寸已成为输入的1/4，接下来是四个阶段，也就是代码中的layer1,
layer2,layer3,layer4。这里用make_layer函数产生四个layer，需要用户输入每个
layer的block数目（即layers列表)以及采用的block类型（基础版还是bottleneck版）
'''


class ResNet(nn.Module):

    def __init__(self, block, layers, last_conv_stride=1, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        '''
        _make_layer第一个输入block是Bottleneck或BasicBlock类，第二个输入是该blocks的输出
        channel，第三个输入是每个blocks中包含多少个residual子结构，因此layers这个列表就是
        resnet50的[3, 4, 6, 3]。
        '''
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 该部分是将每个blocks的第一个residual结构保存在layers列表中。
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


"""
这里比较简单，就是调用上面ResNet对象，输入block类型和block数目，这里可以看到
resnet18和resnet34用的是基础版block，因为此时网络还不深，不太需要考虑模型的
效率，而当网络加深到52，101，152层时则有必要引入bottleneck结构，方便模型的
存储和计算。另外是否加载预训练权重是可选的，具体就是调用model_zoo加载指定链接
地址的序列化文件，反序列化为权重文件。
"""


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 构建网络结构
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    '''
    最后通过调用model的load_state_dict方法用预训练的模型参数来初始化你构建的
    网络结构，这个方法就是PyTorch中通用的用一个模型的参数初始化另一个模型的层
    的操作。load_state_dict方法还有一个重要的参数是strict，该参数默认是True，
    表示预训练模型的层和你的网络结构层严格对应相等（比如层名和维度）。最后通过
    调用model的load_state_dict方法用预训练的模型参数来初始化你构建的网络结构，
    这个方法就是PyTorch中通用的用一个模型的参数初始化另一个模型的层的操作。
    load_state_dict方法还有一个重要的参数是strict，该参数默认是True，表示
    预训练模型的层和你的网络结构层严格对应相等（比如层名和维度）。
    '''
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


# ------------------------------------3构建resnet模型-----------------------------------------
# ------------------------------------4数据预处理-----------------------------------------
path = opt.path


def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0], 'rb') as lbpath:
        # frombuffer将data以流的形式读入转化成ndarray对象
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()  # 读取数据

batch_size = opt.batchsize
num_classes = 10

# 转变颜色空间
X_train = [cv2.cvtColor(cv2.resize(i, (100, 100)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (100, 100)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255

x_train = torch.from_numpy(x_train).cuda()
x_train = x_train.permute(0, 3, 1, 2)
y_train = torch.from_numpy(y_train).long().cuda()
x_test = torch.from_numpy(x_test).cuda()
y_test = torch.from_numpy(y_test).long().cuda()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchsize)

# ------------------------------------4数据预处理-----------------------------------------
# ------------------------------------5调用模型，进行训练-----------------------------------------
model = resnet50()
model = nn.DataParallel(model).cuda()  # gpu并行
cudnn.benchmark = True
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
loss = nn.CrossEntropyLoss()
epochs = opt.epochs
for epoch in range(epochs):
    train_loss = 0.0  # 每个epoch损失初始化
    model.train()  # 训练模式
    for i, data in enumerate(train_dataloader):
        train_pre = model(data[0])  # 预测结果
        batch_loss = loss(train_pre, data[1])  # 计算每批损失
        optimizer.zero_grad()  # 优化器梯度清零
        batch_loss.backward()  # 误差函数反向传播
        optimizer.step()  # 优化器更新
        train_loss += batch_loss.item()  # 计算每个epoch总损失
    print('epoch[%1d/%1d] , loss %8f' % (epoch + 1, epochs, train_loss / len(train_dataloader)))
# ------------------------------------5调用模型，进行训练-----------------------------------------
