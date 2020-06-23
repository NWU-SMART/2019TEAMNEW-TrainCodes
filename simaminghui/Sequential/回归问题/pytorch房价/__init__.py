# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/22 002217:10
# 文件名称：__init__.py
# 开发工具：PyCharm

# ---------------------------------加载的数据包-------------------------------

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------加载数据，处理数据--------------------------
# 数据路径
path = "D:\DataList\\boston\\boston_housing.npz"
# 读取数据
f = np.load(path)
# 404个训练集，102个测试
x_train = f['x'][:404]  # 训练数据，0——403
y_train = f['y'][:404]  # 训练数据目标，0——403

x_valid = f['x'][404:]  # 测试数据，404——结束
y_valid = f['y'][404:]  # 测试数据目标，404——结束
# 与后面处理过后的数据进行对比
print('我是初始：',x_train[0],y_train[0])
# 转换成DataFrame格式,表
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
# 形成对比
print('DataFrame后：',x_train_pd.head(1))  # 前5行数据

# ---------------------------------数据归一化--------------------------
# 使用Min-Max标准化，指对原始数据进行线性变换，将值映射到[0,1]之间。
# 公式为(x-min(X))/(max(X)-min(X))，大X表示所有小x的集合
min_max_scale = MinMaxScaler()

# 训练数据归一化
min_max_scale.fit(x_train_pd)  # 得到最大值和最小值
x_train = min_max_scale.transform(x_train_pd)  # 进行缩放
min_max_scale.fit(y_train_pd)
t_train = min_max_scale.transform(y_train_pd)

# 验证集归一化
min_max_scale.fit(x_valid_pd)
x_valid = min_max_scale.transform(x_valid_pd)
min_max_scale.fit(y_valid_pd)
y_valid = min_max_scale.transform(y_train_pd)
# 和前面形成对比
print(x_train[0])

# -------------在pytorch中，出入网络的参数是tensor格式，所以将数据转换为tensor------------------------
x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
x_valid, y_valid = torch.Tensor(x_valid), torch.Tensor(y_valid)
# 与前面数据进行对比
print(x_train[0])


# ---------------------------------构建网络----------------------------------------
class HouseModel(torch.nn.Module):  # 必须继承
    def __init__(self, feature, output):  # 绑定两个属性，输入的特征数和输出的个数
        super(HouseModel, self).__init__()  # 固定格式
        self.dense1 = torch.nn.Linear(feature, 10)
        self.dense2 = torch.nn.Linear(10, 15)  # 由10转为15
        self.dropout = torch.nn.Dropout(0.2)  # 20%神经元不工作
        self.dense3 = torch.nn.Linear(15, output)  # 15转为output

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)  # 将输出激活
        x = self.dense2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense3(x)


        return x


model = HouseModel(13, 1)  # 模型定义
print(model)

# 优化
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # SGD优化，步长为0.01
# 损失
loss_mse = torch.nn.MSELoss()  # mse损失函数（均方误差）


#-----------------------------------------训练模型------------------------------------
loss_list = [] # 损失列表，每次保存一个loss
# 开始训练，300次,没有分批量。下次写，每个批量参数不同
for t in range(300):
    prediction = model(x_train)

    loss = loss_mse(prediction,y_train) # 预测值和目标值得差距
    loss_list.append(loss)# 将第t轮的训练放入loss集合
    optimizer.zero_grad() # 梯度不会自动清零，所以添加代码
    loss.backward() # 反向传播
    optimizer.step() #更新参数
    print('第{}轮训练,损失为{}'.format(t,loss))

y = model(x_valid)
print('我是预测的',y)
print('我是准确的',y_valid)
plt.plot(loss_list,'r')
plt.show()
