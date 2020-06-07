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
from torch.utils.data import DataLoader,Dataset,TensorDataset
# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理------------------------------------------

f = np.load('mnist.npz')  # 导入数据
print(f.files)
X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)#(60000, 28, 28)
print(X_test.shape)#(10000, 28, 28)

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(X_train.shape[1:])

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))  # (60000,784)
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))  # (10000,784)
X_train = torch.from_numpy(X_train)  # 转为tensor
X_test = torch.from_numpy(X_test)
set=TensorDataset(X_train,X_train)
loader=DataLoader(dataset=set,batch_size=128,shuffle=False)
# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------


class normalize(nn.Module):
    def __init__(self):
        super(normalize, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        self.relu1 = nn.ReLU()
        self.normalize=nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 784)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out=self.normalize(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


model = normalize()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.MSELoss()

# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------
loss_total = []
epoch_total = []

epochs = 5
for epoch in range(epochs):
    train_loss=0.0
    for i,data in enumerate(loader):
        pre = model(data[0])
        train_loss = loss(pre, data[1])
        loss_total.append(train_loss)
        epoch_total.append(epoch)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss+=train_loss
    print('epoch %3d, loss %10f' % (epoch, train_loss/len(loader)))

# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5可视化------------------------------------------
# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
pre = model(X_test).detach().numpy()
n=10
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
