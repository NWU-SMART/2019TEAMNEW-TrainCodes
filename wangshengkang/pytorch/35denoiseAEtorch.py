# -*- coding: utf-8 -*-
# @Time: 2020/6/8 20:57
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
path = 'mnist.npz'
f = np.load(path)
print(f.files)

X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)  # (60000, 28, 28)
print(X_test.shape)  # (10000, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # 60000*28*28*1
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)  # 10000*28*28*1

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

# 参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
# 参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
# 参数size(int 或者整数元组)：输出的值赋在shape里，默认为None。
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

# 把噪声限制在0,1之间，小于0的为0，大于1的为1
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).float()  # 转为tensor，改变位置，转换类型
X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
X_train_noisy = torch.from_numpy(X_train_noisy).permute(0, 3, 1, 2).float()
X_test_noisy = torch.from_numpy(X_test_noisy).permute(0, 3, 1, 2).float()

X_train_dataset = TensorDataset(X_train_noisy, X_train)
X_test_dataset = TensorDataset(X_test_noisy, X_test)
X_train_loader = DataLoader(X_train_dataset, batch_size=128)
X_test_loader = DataLoader(X_test_dataset, batch_size=128)


# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------


class denoiseAE(nn.Module):
    def __init__(self):
        super(denoiseAE, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),  # 28*28*32
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 14*14*32
            nn.Conv2d(32, 32, (3, 3), padding=1),  # 14*14*32
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 7*7*32

            nn.Conv2d(32, 32, (3, 3), padding=1),  # 7*7*32
            nn.ReLU(),
            nn.Upsample((14, 14)),  # 14*14*32
            nn.Conv2d(32, 32, (3, 3), padding=1),  # 14*14*32
            nn.ReLU(),
            nn.Upsample((28, 28)),  # 28*28*32
            nn.Conv2d(32, 1, (3, 3), padding=1),  # 28*28*1
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.net(x)
        return out


model = denoiseAE()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.BCELoss()

# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------
loss_total = []
epoch_total = []

epochs = 3
for epoch in range(epochs):
    train_loss = 0
    for i, data in enumerate(X_train_loader):
        pre = model(data[0])  # 预测结果
        batch_loss = loss(pre, data[1])  # 计算损失
        optimizer.zero_grad()  # 优化器梯度清零
        batch_loss.backward()  # 损失反向传播
        optimizer.step()  # 优化器更新
        train_loss += batch_loss.item()  # 同一个epoch里的所有batch损失加起来
        print('batch[{}/{}], loss{:.5f}'.format(i + 1, len(X_train_loader), train_loss))
    print('epoch[{}/{}], loss{:.5f}'.format(epoch + 1, epochs, train_loss / len(X_train_loader)))
    loss_total.append(train_loss / len(X_train_loader))
    epoch_total.append(epoch)
# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5可视化------------------------------------------

# RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
pre_test = model(X_test_noisy).detach().numpy()

n = 10
plt.figure(figsize=(20, 6))
for i in range(10):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)  # 隐藏坐标轴
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(pre_test[i].reshape(28, 28))
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
