# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年5月27日
# 开发内容 房价预测的pytorch实现，使用class继承的方法
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


#   ---------------------- 数据的载入和处理 ----------------------------
path = 'boston_housing.npz'
f = np.load(path)      # numpy.load（）读取数据
# 404个训练，102个测试
# 训练数据
x_train = f['x'][:404]  # 下标0到下标403
y_train = f['y'][:404]
# 测试数据
x_valid = f['x'][404:]  # 下标404到下标505
y_valid = f['y'][404:]
f.close()   # 关闭文件

# 转成DataFrame格式方便数据处理    DataFrame格式可理解为一张表
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)



#  -------------------------- 数据归一化处理 -------------------------------
# 训练集归一化 归一化可以减少量纲不同带来的影响，使得不同特征之间具有可比性；
# 这里用的是线性归一化，公式是(x-xMin)/(xMax-xMin)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)
min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

# 在pytorch中，传入网络内的参数必须是tensor格式，所以在这里先要将数据全部转换成tensor
# 四组数据
x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
x_test, y_test = torch.Tensor(x_valid), torch.Tensor(y_valid)


#   ---------------------- 构建模型 ---------------------------
class HouseModel(torch.nn.Module):  # 继承torch.nn.Module
    def __init__(self, feature, output):  # 绑定两个属性，输入的特征数，输出的个数
        super(HouseModel, self).__init__()
        self.dense1 = torch.nn.Linear(feature, 10)  # 三个全连接层一个dropout层
        self.dense2 = torch.nn.Linear(10, 15)
        self.dropout = torch.nn.Dropout(0.2)
        self.dense3 = torch.nn.Linear(15, output)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = F.relu(x)
        x = self. dropout(x)
        x = self.dense3(x)
      # x = F.Linear(x) # F.Linear找不到，在网上也没找到具体的用法，

        return x


model = HouseModel(x_train_pd.shape[1], 1)  # 实例化房屋模型
print(model)  # 在这里看一下模型的结构，可以发现in_features=13，out_features=1

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器为SGD，学习率是1e-3
loss_func = torch.nn.MSELoss()   # 定义损失函数为均方误差


#   ---------------------- 训练模型 ---------------------------
loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(300):
    train_prediction = model(x_train)
    loss = loss_func(train_prediction, y_train)  # 计算损失
    loss_list.append(loss) # 使用append()方法把每一次的loss添加到loss_list中

    optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
    loss.backward()  # 反向传播，计算参数
    optimizer.step()  # 更新参数
    print(loss)


plt.plot(loss_list, 'r-')
plt.show()


