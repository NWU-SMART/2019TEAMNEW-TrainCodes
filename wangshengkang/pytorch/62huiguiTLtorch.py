# -*- coding: utf-8 -*-
# @Time: 2020/6/18 13:14
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
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd

# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理-----------------------------------------
plt.switch_backend('agg')  # 服务器没有gui
# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='torch')
# 调用add_argument()方法添加参数
parser.add_argument('--path', default='mnist.npz', type=str, help='the path to dataset')
parser.add_argument('--batchsize', default='32', type=int, help='batchsize')
parser.add_argument('--gpu', default='6', type=str, help='choose which gpu to use')
# 使用parse_args()解析添加的参数
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu  # 选择gpu

path = opt.path
f = np.load(path)
x_train = f['x_train']
x_test = f['x_test']
y_train = f['y_train']
y_test = f['y_test']
f.close()

x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]


def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]
    return np.round(price, 2)


y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]

print(len(y_train))
print(len(y_test))

x_train = torch.from_numpy(x_train).float().cuda()
x_train = x_train.permute(0, 3, 1, 2)
y_train = torch.from_numpy(y_train).float().cuda()
x_test = torch.from_numpy(x_test).float().cuda()
y_test = torch.from_numpy(y_test).float().cuda()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchsize)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchsize)

# 调用vgg模型，progress=True，显示进度条
vggmodel = torchvision.models.vgg16(pretrained=True,progress=True)
# 固定vgg的参数
for param in vggmodel.parameters():
    param.requires_grad = False


class huigui(nn.Module):
    def __init__(self):
        super(huigui, self).__init__()

        # 更换vgg最后三个全连接层
        vggmodel.classifier = nn.Sequential(
            nn.Flatten(),  # 512*7*7-->25088
            nn.Linear(25088, 256),  # 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),  # 10
            )
        self.vgg = vggmodel

    def forward(self, x):
        x = self.vgg(x)
        return x


model = huigui().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
loss = nn.MSELoss()

epochs=5
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
