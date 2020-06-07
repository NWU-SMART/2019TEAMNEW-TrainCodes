# -*- coding: utf-8 -*-
# @Time: 2020/6/3 18:21
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
plt.switch_backend('agg')  # 服务器没有gui

path = 'mnist.npz'
f = np.load(path)
print(f.files)

X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)  # (60000, 28, 28)
print(X_test.shape)  # (10000, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

X_train = torch.from_numpy((X_train))
X_test = torch.from_numpy((X_test))

print('X_train shape', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------


class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)  # 16*28*28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))  # 16*14*14
        self.conv2 = nn.Conv2d(16, 8, (3, 3), padding=1)  # 8*14*14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2, 2), padding=1)  # 8*7*7
        self.conv3 = nn.Conv2d(8, 8, (3, 3), padding=1)  # 8*7*7
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d((2, 2), padding=1)  # 8*4*4

        self.conv4 = nn.Conv2d(8, 8, (3, 3), padding=1)  # 8*4*4
        self.relu4 = nn.ReLU()
        self.up1 = nn.Upsample((8, 8))  # 8*8*8
        self.conv5 = nn.Conv2d(8, 8, (3, 3), padding=1)  # 8*8*8
        self.relu5 = nn.ReLU()
        self.up2 = nn.Upsample((16, 16))  # 8*16*16
        self.conv6 = nn.Conv2d(8, 16, (3, 3))  # 16*14*14
        self.up3 = nn.Upsample((28, 28))  # 16*28*28
        self.conv7 = nn.Conv2d(16, 1, (3, 3), padding=1)  # 1*28*28
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.up1(out)
        out = self.conv5(out)
        out = self.relu5(out)
        out = self.up2(out)
        out = self.conv6(out)
        out = self.up3(out)
        out = self.conv7(out)
        out = self.sigmoid(out)
        return out


model = cnn()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.BCELoss()

# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------
loss_total = []
epoch_total = []

epochs = 30
for epoch in range(epochs):
    pre = model(X_train)
    train_loss = loss(pre, X_train)
    loss_total.append(train_loss)
    epoch_total.append(epoch)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    print('epoch %3d, loss %10f' % (epoch, train_loss))
# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5可视化------------------------------------------

# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
pre = model(X_test).detach().numpy()

n = 10
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
plt.savefig(fname='33tu1.png')  # 保存图片
plt.show()

plt.plot(epoch_total, loss_total, label='loss')
plt.title('torch loss')  # 题目
plt.xlabel('Epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train'], loc='upper left')  # 图线示例
plt.savefig(fname='33tu2.png')
plt.show()  # 画图

# ------------------------------------5可视化------------------------------------------
