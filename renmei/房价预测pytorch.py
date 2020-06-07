#-----------------------------------------------------
#----------------------任梅------------------------
#-------------------------2020.05.27---------------------
# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练

# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn  as nn
import torch

from torchvision import transforms
#  -------------------------- 1、导入需要包 -------------------------------



#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
# 数据在服务器可以访问
#train_data.shape:(404, 13),test_data.shape:(102, 13),
#train_targets.shape:(404,),test_targets.shape:(102,)
#the data compromises 13 features
#(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()  # 加载数据（国外服务器无法访问）

# 数据放到本地路径
# D:\\keras_datasets\\boston_housing.npz(本地路径)
path = 'D:\\keras_datasets\\boston_housing.npz'
f = np.load(path)
# 404个训练，102个测试
# 训练数据
x_train=f['x'][:404]  # 下标0到下标403
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 下标404到下标505
y_valid=f['y'][404:]
f.close()
# 数据放到本地路径

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)
#  -------------------------- 2、房价训练和测试数据载入 -------------------------------


#  -------------------------- 3、数据归一化 -------------------------------
# 训练集归一化
def MinMaxScaler(data):
    min=np.amin(data)
    max=np.amax(data)
    return (data-min)/(max-min)
x_train=MinMaxScaler(x_train_pd)
y_train = MinMaxScaler(y_train_pd)

# 验证集归一化
x_valid=MinMaxScaler(x_valid_pd)
y_valid=MinMaxScaler(y_valid)
#  -------------------------- 3、数据归一化  -------------------------------

#  -------------------------- 4、模型训练

epoch=5
lr=0.03
net=nn.Sequential(
    nn.Linear(x_train_pd.shape[1],10),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(10,15),
    nn.ReLU(),
    nn.Linear(10,1),
    nn.Linear()
)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.adam(net.parameters(),lr=lr)
for i in range(epoch):
    out=net(x_train)
    loss=loss(out,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    num_correct=0
    out2=net(x_valid)
    predict=torch.max(out2,1)
    num_correct+=(predict==y_valid).sum()
    accuracy=num_correct.numpy()/len(x_valid)
    print("第%d次迭代。准确率为%f"%(epoch+1,accuracy))