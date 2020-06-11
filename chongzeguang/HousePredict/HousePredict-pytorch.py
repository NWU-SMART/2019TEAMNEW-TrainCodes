# ----------------开发者信息--------------------------------#
# 开发者：崇泽光
# 开发日期：2020年6月11日
# 修改日期：
# 修改人：
# 修改内容：

# 导入所需要的包
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 读取数据
path = 'D:\\keras_datasets\\boston_housing.npz'
f = np.load(path)

# 设定训练集，测试集
x_train = f['x'][:404] # 下标0到下标403
y_train = f['y'][:404]
x_valid = f['x'][404:] # 下标404到下标505
y_valid = f['y'][404:]
f.close()

# 转换数据结构，方便数据处理（DataFrame）
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)

# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
x_train = torch.Tensor(x_train) #Tensor是PyTorch中重要的数据结构，可认为是一个高维数组。
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)
y_train = torch.Tensor(y_train) #Tensor是PyTorch中重要的数据结构，可认为是一个高维数组。

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)
x_valid = torch.Tensor(x_valid)
min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)
y_valid = torch.Tensor(y_valid)


# 构建模型
input_size = x_train_pd.shape[1]  #输入

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, 10), #设置输入和输出大小
            torch.nn.ReLU(),  #relu激活
            torch.nn.Dropout(0.2), #dropout防止过拟合
            torch.nn.Linear(10, 15), #对传入数据应用线性变换
            torch.nn.ReLU(),
            torch.nn.Linear(15, 1) #输入大小为15，输出大小为1
        )

    def forward(self, x): #前向传播函数forward
        x = self.layer(x)
        return x

model = MLP()
print(model)

# 训练参数设置
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  #优化器，lr (float, 可选)是学习率（默认：1e-3）
loss_fn = torch.nn.MSELoss() #损失函数，选定均方损失函数
Epoch = 200

# 训练模型
for i in range(Epoch):
    x = model(x_train)  # 向前传播
    loss = loss_fn(x, y_train)  # 计算损失
    optimizer.zero_grad() # 梯度清零
    loss.backward() # 反向传播
    optimizer.step() # 更新梯度

    if (i + 1) % 1 == 0:
        print(loss.item()) # 每训练一个epoch，打印一次损失值