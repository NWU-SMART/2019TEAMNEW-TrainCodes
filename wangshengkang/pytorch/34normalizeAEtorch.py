# -*- coding: utf-8 -*-
# @Time: 2020/6/4 21:53
# @Author: wangshengkang
# -----------------------------------代码布局--------------------------------------------
# 1引入pytorch，numpy，matplotlib，IPython等包
# 2导入数据，数据预处理
# 3建立模型
# 4训练模型，预测结果
# 5结果以及损失函数可视化
# -----------------------------------代码布局--------------------------------------------
# ------------------------------------1引入包-----------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理------------------------------------------

f = np.load('mnist.npz')  # 导入数据
print(f.files)
X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)  # (60000, 28, 28)
print(X_test.shape)  # (10000, 28, 28)

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_train.shape[1:])  # (28, 28)

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))  # (60000,784)
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))  # (10000,784)
X_train = torch.from_numpy(X_train)  # 转为tensor
X_test = torch.from_numpy(X_test)
set = TensorDataset(X_train, X_train)  # 将数据集包装为TensorDataset
loader = DataLoader(dataset=set, batch_size=128, shuffle=False)  # 使用dataloader类


# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------


class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        self.fc1 = nn.Linear(784, 32)  # 32
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 784)  # 784
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


model = normalize()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.MSELoss()

# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------
loss_total = []  # 放置损失函数的列表
epoch_total = []

epochs = 5
for epoch in range(epochs):
    noregular_loss = 0.0
    train_loss = 0.0  # 初始化损失函数的值
    for i, data in enumerate(loader):  # 挨个batch训练
        regularization_loss = 0
        for param in model.parameters():  # l1正则化 L1范数是参数矩阵W中元素的绝对值之和
            regularization_loss += torch.sum(torch.abs(param))
        pre = model(data[0])  # data[0]为训练图像
        batch_loss = loss(pre, data[1])  # data[1]为真实图像
        loss_regularize = batch_loss + 0.000001 * regularization_loss
        optimizer.zero_grad()  # 梯度清零
        loss_regularize.backward()  # 损失函数反向传播
        optimizer.step()  # 优化器更新
        noregular_loss += batch_loss
        train_loss += loss_regularize  # 一个epoch内的所有loss加起来
    print('epoch %3d, noregular_loss %10f, train_loss %10f' % (
        epoch+1, noregular_loss / len(loader), train_loss / len(loader)))
    loss_total.append(train_loss / len(loader))  # 每个epoch的loss加进来
    epoch_total.append(epoch)

# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5可视化------------------------------------------
# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
pre = model(X_test).detach().numpy()
n = 10
plt.figure(figsize=(20, 6))
for i in range(10):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))  # 打印真实图片
    plt.gray()
    ax.get_xaxis().set_visible(False)  # 隐藏坐标轴
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(pre[i].reshape(28, 28))  # 打印模型解码的图片
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 画出loss曲线
plt.plot(epoch_total, loss_total, label='loss')
plt.title('torch loss')  # 题目
plt.xlabel('Epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train'], loc='upper left')  # 图线示例
plt.show()  # 画图

# ------------------------------------5可视化------------------------------------------
