# -*- coding: utf-8 -*-
# @Time: 2020/6/3 10:38
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
# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理------------------------------------------

f = np.load('mnist.npz')  # 导入数据
print(f.files)
X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)
print(X_test.shape)

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))  # (60000,784)
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))  # (10000,784)
X_train = torch.from_numpy(X_train)  # 转为tensor
X_test = torch.from_numpy(X_test)

# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------


class multi(nn.Module):
    def __init__(self):
        super(multi, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)

        return out


model = multi()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.MSELoss()

# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------
loss_total = []
epoch_total = []

epochs = 20
for epoch in range(epochs):
    pre = model(X_train)
    train_loss = loss(pre, X_train)
    loss_total.append(train_loss)
    epoch_total.append(epoch)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    print('epoch %3d, loss %10f' % (epoch, train_loss))


class multi2(nn.Module):
    def __init__(self):
        super(multi2, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)

        return out


model2 = multi2()
# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
yasuo = model2(X_test).detach().numpy()

# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5可视化------------------------------------------
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(yasuo[i].reshape(4, 16).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)  # 隐藏坐标轴
    ax.get_yaxis().set_visible(False)
plt.show()
# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
pre = model(X_train).detach().numpy()

plt.figure(figsize=(20, 6))
for i in range(10):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(pre[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

plt.plot(epoch_total, loss_total, label='loss')
plt.title('torch loss')  # 题目
plt.xlabel('Epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train'], loc='upper left')  # 图线示例
plt.show()  # 画图

# ------------------------------------5可视化------------------------------------------
