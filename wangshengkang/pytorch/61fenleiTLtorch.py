# -*- coding: utf-8 -*-
# @Time: 2020/6/15 11:12
# @Author: wangshengkang
# -----------------------------------代码布局--------------------------------------------
# 1引入keras，numpy，matplotlib，IPython等包
# 2导入数据，数据预处理
# 3建立模型
# 4训练模型
# 5保存模型
# 6画出准确率和损失函数的变化曲线
# -----------------------------------代码布局--------------------------------------------
# ------------------------------------1引入包-----------------------------------------------
import gzip
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理-----------------------------------------
plt.switch_backend('agg')  # 服务器没有gui
# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='denoiseAE')
# 调用add_argument()方法添加参数
parser.add_argument('--path', default='./', type=str, help='the path to dataset')
parser.add_argument('--batchsize', default='32', type=int, help='batchsize')
parser.add_argument('--gpu', default='6', type=str, help='choose which gpu to use')
# 使用parse_args()解析添加的参数
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu  # 选择gpu

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
epochs = 5

# 转变颜色空间
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

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
train_dataloader = DataLoader(train_dataset, batch_size=128)
test_dataloader = DataLoader(test_dataset, batch_size=128)

# 调用vgg模型
vggmodel = torchvision.models.vgg16(pretrained=True, progress=True)
# 固定vgg的参数
for param in vggmodel.parameters():
    param.requires_grad = False


class fenlei(nn.Module):
    def __init__(self):
        super(fenlei, self).__init__()

        # 更换vgg最后三个全连接层
        vggmodel.classifier = nn.Sequential(
            nn.Flatten(),  # 512*7*7-->25088
            nn.Linear(25088, 256),  # 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),  # 10
            nn.Softmax())
        self.vgg = vggmodel

    def forward(self, x):
        x = self.vgg(x)
        return x


model = fenlei().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
loss = nn.CrossEntropyLoss()

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
