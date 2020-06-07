# ------------------------作者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/23 20:04
# @Author: wangshengkang

# --------------------------作者信息--------------------------------------
# ----------------------代码布局-------------------------------------
# 1.引入keras，matplotlib，numpy，sklearn，pandas包
# 2.导入数据
# 3.数据归一化
# 4.模型建立
# 5.损失函数可视化
# 6.预测结果
# ---------------------------------------------------------------------

# --------------------------1引入相关包-----------------------------
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --------------------------1引入相关包---------------------------

# --------------------------2导入数据----------------------------
data = np.load('boston_housing.npz')  # 导入数据
train_x = data['x'][:404]  # 前404个数据为训练集
train_y = data['y'][:404]
valid_x = data['x'][404:]  # 后面的数据为验证集
valid_y = data['y'][404:]

# 讲数据转换为DataFrame的形式
train_x_pd = pd.DataFrame(train_x)
train_y_pd = pd.DataFrame(train_y)
valid_x_pd = pd.DataFrame(valid_x)
valid_y_pd = pd.DataFrame(valid_y)

# 打印训练集前5个，训练集形状
print(train_x_pd.head(5))
print(train_y_pd.head(5))
print(train_x_pd.shape)
print(train_x_pd.shape[1])
# ----------------------------2导入数据----------------------------

# ----------------------------3数据归一化----------------------------
min_max_scale = MinMaxScaler()
min_max_scale.fit(train_x_pd)  # fit用来获取min，max
train_x_sc = min_max_scale.transform(train_x_pd)  # transform根据min，max来scale
train_x_te = torch.autograd.Variable(torch.from_numpy(train_x_sc))  # 转化为tensor形式
train_x_te = train_x_te.float()  # double转变为float

min_max_scale.fit(train_y_pd)
train_y_sc = min_max_scale.transform(train_y_pd)
train_y_te = torch.autograd.Variable(torch.from_numpy(train_y_sc))
train_y_te = train_y_te.float()

min_max_scale.fit(valid_x_pd)
valid_x_sc = min_max_scale.transform(valid_x_pd)
valid_x_te = torch.autograd.Variable(torch.from_numpy(valid_x_sc))
valid_x_te = valid_x_te.float()

min_max_scale.fit(valid_y_pd)
valid_y_sc = min_max_scale.transform(valid_y_pd)
valid_y_te = torch.autograd.Variable(torch.from_numpy(valid_y_sc))
valid_y_te = valid_y_te.float()


# -------------------------3数据归一化------------------------------

# ----------------------------4建立模型-------------------------------
# 类方法1-4
class house(nn.Module):
    def __init__(self):
        super(house, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(13, 10),  # 全连接层
            nn.ReLU(),  # relu激活函数
            nn.Dropout(0.2),  # Dropout

            nn.Linear(10, 15),
            nn.ReLU(),

            nn.Linear(15, 1),
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class house2(nn.Module):
    def __init__(self):
        super(house2, self).__init__()
        self.fc1 = nn.Linear(13, 10)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(10, 15)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        return self.fc3(self.relu2(self.fc2(x)))


class house3(nn.Module):
    def __init__(self):
        super(house3, self).__init__()
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(13, 10)),
            ('relu1', nn.ReLU()),
            ('dp', nn.Dropout(0.2)),
            ('fc2', nn.Linear(10, 15)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(15, 1))
        ]))

    def forward(self, x):
        return self.mlp(x)


class house4(nn.Module):
    def __init__(self):
        super(house4, self).__init__()
        self.mlp = nn.Sequential()
        self.mlp.add_module('fc1', nn.Linear(13, 10))
        self.mlp.add_module('relu1', nn.ReLU())
        self.mlp.add_module('dp', nn.Dropout(0.2))
        self.mlp.add_module('fc2', nn.Linear(10, 15))
        self.mlp.add_module('relu2', nn.ReLU())
        self.mlp.add_module('fc3', nn.Linear(15, 1))

    def forward(self, x):
        return self.mlp(x)


# 序列化模型方法1-3
# model = nn.Sequential()
# model.add_module('fc1', nn.Linear(13, 10))
# model.add_module('relu1', nn.ReLU())
# model.add_module('dp', nn.Dropout(0.2))
# model.add_module('fc2', nn.Linear(10, 15))
# model.add_module('relu2', nn.ReLU())
# model.add_module('fc3', nn.Linear(15, 1))


# model = nn.Sequential(
#             nn.Linear(13, 10),  # 全连接层
#             nn.ReLU(),  # relu激活函数
#             nn.Dropout(0.2),  # Dropout
#             nn.Linear(10, 15),
#             nn.ReLU(),
#             nn.Linear(15, 1)
# )


# model = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(13, 10)),
#             ('relu1', nn.ReLU()),
#             ('dp', nn.Dropout(0.2)),
#             ('fc2', nn.Linear(10, 15)),
#             ('relu2', nn.ReLU()),
#             ('fc3', nn.Linear(15, 1))
#         ]))


model = house()  # 用类来创建模型，如果用序列化模型方法，需要注释掉此行
loss = nn.MSELoss(reduction='sum')  # mse损失，计算方式采用sum（默认为mean）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
epochs = 200

iteration = []  # list存放epoch数
loss_total = []  # list存放损失
for epoch in range(epochs):
    train_loss = 0.0
    model.train()  # 训练模式
    train_pre = model(train_x_te)
    batch_loss = loss(train_y_te, train_pre)  # 损失函数
    iteration.append(epoch)  # 将epoch放到list中
    loss_total.append(batch_loss)  # 将loss放到list中
    optimizer.zero_grad()  # 优化器梯度清零
    batch_loss.backward()  # 损失函数反向传播
    optimizer.step()  # 梯度，更新参数值
    print('epoch %3d , loss %3d' % (epoch, batch_loss))

# ------------------------------4建立模型------------------------------

# -------------------------------5损失函数可视化------------------------
plt.plot(iteration, loss_total, label="loss")  # iteration和loss对应画出
plt.title('torch loss')  # 题目
plt.xlabel('Epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train'], loc='upper left')  # 图线示例
plt.show()  # 画图

# --------------------------------5损失函数可视化------------------------

# -------------------------------6保存模型并预测--------------------

torch.save(model.state_dict(), "housetorch.pth")  # 保存模型参数
model.load_state_dict(torch.load('housetorch.pth'))  # 加载模型
model.eval()  # 评估模式
valid_pre = model(valid_x_te)  # 预测结果

# 为了解决RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
valid_pre_numpy = valid_pre.detach().numpy()
min_max_scale.fit(valid_y_pd)
valid_pre_fg = min_max_scale.inverse_transform(valid_pre_numpy)  # 反归一化
print(valid_pre_fg)  # 打印预测结果

# ----------------------------------6保存模型并预测---------------
