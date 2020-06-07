# ----------------开发者信息--------------------------------#
# 开发者：姜媛
# 开发日期：2020年5月27日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#


#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torchvision
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn as nn

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
path = 'C:\\Users\\HP\\Desktop\\boston_housing.npz'  # 路径
f = np.load(path)  # numpy.load（）读取数据
# 404个训练，102个测试
# 训练数据
x_train = f['x'][:404]  # 下标0到下标403
y_train = f['y'][:404]
# 测试数据
x_valid = f['x'][404:]  # 下标404到下标505
y_valid = f['y'][404:]
f.close()  # 关闭文件

# 转成DataFrame格式
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)
#  -------------------------- 2、房价训练和测试数据载入 -------------------------------


#  -------------------------- 3、数据处理 -------------------------------
# 训练集归一化 pytorch框架需要进行Tensor
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = torch.Tensor(min_max_scaler.transform(x_train_pd))

min_max_scaler.fit(y_train_pd)
y_train = torch.Tensor(min_max_scaler.transform(y_train_pd))

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = torch.Tensor(min_max_scaler.transform(x_valid_pd))

min_max_scaler.fit(y_valid_pd)
y_valid = torch.Tensor(min_max_scaler.transform(y_valid_pd))
#  -------------------------- 3、数据处理 ------------------------------


#  -------------------------- 4、Sequential模型训练   -------------------------------
inputs = x_train_pd.shape[1]  # 输入
class HousePredict(nn.Module):
    def __init__(self):
        super(HousePredict, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(inputs, 10),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(10, 15),
            nn.ReLU(True),
            nn.Linear(15, 1),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


model = HousePredict()  # 定义模型
print(model)  # 打印模型
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam优化器
loss_fn = torch.nn.MSELoss()  # 损失函数

epochs = 500
iteration = []  # list存放epoch数
loss_list = []  # list存放损失
for epoch in range(500):  # 迭代次数
    loss = 0.0
    y_valid_pd = model(x_train)  # 向前传播 y的预测值
    loss = loss_fn(y_train,y_valid_pd)  # 计算预测和真实值的损失
    iteration.append(epoch)  # 将epoch放到list中
    loss_list.append(loss)  # 将loss放到list中
    print(loss) # 显示损失
    optimizer.zero_grad()  # 优化器梯度清零
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新梯度
#  -------------------------- 4、Sequential模型训练    -------------------------------


# -------------------------------5损失函数可视化------------------------
plt.plot(iteration, loss_list, label="loss")  # iteration和loss对应画出
plt.title('torch loss')  # 题目
plt.xlabel('Epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train'], loc='upper left')  # 图线示例
plt.show()  # 画图
# --------------------------------5损失函数可视化------------------------