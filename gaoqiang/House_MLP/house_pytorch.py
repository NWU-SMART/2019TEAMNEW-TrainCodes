# ----------------开发者信息--------------------------------#
# 开发者：高强
# 开发日期：2020年5月22日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------------代码布局---------------------------#
# 1、导入 pytorch相关包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练

import numpy as np

# --------------------------载入数据集-------------------#
path = 'F:\\Keras代码学习\\keras\\keras_datasets\\boston_housing.npz'  # 路径
f = np.load(path)  # 载入

# ---------------对数据进行预处理--------------------#
# 划分训练集和测试集（404个做训练，102个做测试，一共13个特征）
x_train = f['x'][:404]
y_train = f['y'][:404]
x_test = f['x'][404:]
y_test = f['y'][404:]
f.close()

# 转换成DataFrame数据
# （DataFrame是一个表格型的数据类型，每列值类型可以不同，是最常用的pandas对象。）
import pandas as pd

x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_test_pd = pd.DataFrame(x_test)
y_test_pd = pd.DataFrame(y_test)
# 在用Pandas读取数据之后，往往想要观察一下数据读取是否准确，这就要用到Pandas
# 里面的head( )函数，head( )函数默认只能读取前五行数据。
print(x_train_pd.head())
print('-------------------')
print(y_train_pd.head())

# -------------数据归一化----------#
# MinMaxScaler：归一到 [ 0，1 ] ；MaxAbsScaler：归一到 [ -1，1 ]
from sklearn.preprocessing import MinMaxScaler
import torch

# 训练集
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
x_train = torch.Tensor(x_train)  # 一种包含单一数据类型元素的多维矩阵
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)
y_train = torch.Tensor(y_train)

# 测试集
min_max_scaler.fit(x_test_pd)
x_test = min_max_scaler.transform(x_test_pd)
x_test = torch.Tensor(x_test)
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)
y_test = torch.Tensor(y_test)

import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = x_train_pd.shape[1]


################################  方法一：自定义class      ############################################
# class Model(nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.linear1 = torch.nn.Linear(input_size,10)
#         self.relu1 = torch.nn.ReLU()
#         self.Dropout = torch.nn.Dropout(0.2)
#         self.linear2 = torch.nn.Linear(10,15)
#         self.relu2 = torch.nn.ReLU()
#         self.linear3 = torch.nn.Linear(15,1)

#     def forward(self,x):
#         x = self.linear1(x)
#         x = self.relu1(x)
#         x = self.linear2(x)
#         x = self.relu2(x)
#         x = self.linear3(x)

#         return x


################################  方法二：Sequential ：有序的容器     ############################################
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 10),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(10, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


###############################################################################################################

model = Model()

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
Epoch = 200

## 开始训练 ##
for t in range(Epoch):

    x = model(x_train)  # 向前传播
    loss = loss_fn(x, y_train)  # 计算损失
    # 显示损失
    if (t + 1) % 1 == 0:
        print(loss.item())
    # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 更新梯度
    optimizer.step()

