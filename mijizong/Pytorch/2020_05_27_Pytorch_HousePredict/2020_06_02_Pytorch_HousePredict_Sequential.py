# ----------------------开发者信息-----------------------------------------
 # -*- coding: utf-8 -*-
 # @Time: 2020/6/2
 # @Author: MiJizong
 # @Version: 1.0
 # @FileName: 1.0.py
 # @Software: PyCharm
 # ----------------------开发者信息-----------------------------------------


# ----------------------   代码布局： --------------------------------------
# 1、导入 torch、Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测
# ----------------------   代码布局： --------------------------------------


#  -------------------------- 1、导入需要包 --------------------------------
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 --------------------------------


#  -------------------------- 2、房价训练和测试数据载入 ---------------------

path = 'D:\\Office_software\\PyCharm\\keras_datasets\\boston_housing.npz'
f = np.load(path)  # 读取上面路径的数据
# 404个数据用于训练，102个数据用于测试
# 训练数据
x_train=f['x'][:404]  # 下标0到下标403
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 下标404到下标505
y_valid=f['y'][404:]
f.close()   # 关闭文件

# 将数据转成DataFrame格式
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)
#  -------------------------- 2、房价训练和测试数据载入 ----------------------


#  ------------------------------ 3、数据归一化 -----------------------------
# 训练集归一化
min_max_scaler = MinMaxScaler()  #归一化到 [ 0，1 ]
min_max_scaler.fit(x_train_pd)
x_train = torch.Tensor(min_max_scaler.transform(x_train_pd))

min_max_scaler.fit(y_train_pd)
y_train = torch.Tensor(min_max_scaler.transform(y_train_pd))

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = torch.Tensor(min_max_scaler.transform(x_valid_pd))

min_max_scaler.fit(y_valid_pd)
y_valid = torch.Tensor(min_max_scaler.transform(y_valid_pd))
#  -------------------------- 3、数据归一化  --------------------------------------


#  -------------------------- 4.1、Sequential模型训练   -------------------------------
inputs = x_train_pd.shape[1] #原始数据
class HousePredict1(torch.nn.Module):
    def __init__(self):
        super(HousePredict1, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(inputs, 10),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(10, 15),
            torch.nn.ReLU(),
            torch.nn.Linear(15, 1)
        )

    def forward(self,x):
        x = self.dense(x)
        return x
#  -------------------------- 4.1、Sequential模型训练   -------------------------------

#  -------------------------- 4.2、another method训练    -------------------------------
inputs = x_train_pd.shape[1] #原始数据
class HousePredict2(torch.nn.Module):
    def __init__(self):
        super(HousePredict2, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('dense1', torch.nn.Linear(inputs, 10))
        self.conv.add_module('dropout', torch.nn.Dropout(0.2))
        self.conv.add_module('dense2', torch.nn.Linear(10, 15))
        self.conv.add_module('dense3', torch.nn.Linear(15, 1))

    def forward(self, x):
        x = self.conv(x)
        return x
#  -------------------------- 4.2、another method训练    -------------------------------


#  -------------------------- 5、模型训练 -----------------------------------------
model = HousePredict1()
print(model)

loss_func = torch.nn.MSELoss()                        # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)  # SGD优化器

loss_list = []                                           # 存放loss
loss_item = []                                          # 记下每个epoch
EPOCH = 6000
for epoch in range(EPOCH):
    prediction = model(x_train)
    loss = loss_func(prediction,y_train)               # 计算损失
    loss_item.append(epoch)
    loss_list.append(loss)                              # 将每一次的loss添加到loss_list中
    optimizer.zero_grad()                               # 梯度归零
    loss.backward()                                     # 反向传播
    optimizer.step()                                    # 梯度更新
    print(epoch)
    print(loss)
#  -------------------------- 5、模型训练 -----------------------------------------


#  -------------------------- 6、模型可视化 ---------------------------------------
plt.plot(loss_item, loss_list, label="Train loss")
plt.plot()
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')  # 给图加上图例
plt.savefig('test.png')                      # 保存
plt.show()
#  ---------------------------- 6、模型可视化 ---------------------------------------