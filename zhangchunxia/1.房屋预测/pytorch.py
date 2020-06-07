# ----------------开发者信息---------------------------------
# 开发者：张春霞
# 开发日期：2020年5月29日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------------

# ----------------代码布局------------------------------------
# 1、导入 pytorch相关包
# 2、房价训练数据导入
# 3、数据归一化预处理
# 4、模型构建
# 5、模型训练
# ----------------代码布局-------------------------------------

#  -------------- 1、导入 pytorch相关包 -------------------------------

import numpy as np
import pandas as pd
from keras import Input
from networkx.drawing.tests.test_pylab import plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
#  -------------- 1、导入 pytorch相关包 -------------------------------

#  -------------- 2、房价训练数据导入 ----------------------------------
path = 'D:/northwest/小组视频/1房屋预测/boston_housing.npz'
f = np.load(path)
#划分训练集和测试集
x_train = f['x'][:404]
y_train = f['y'][:404]
x_text = f['x'][404:]
y_text = f['x'][404:]
f.close()
#将数据转换为DataFrame格式,他是一个表格型的数据类型，是panda的对象
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_text_pd = pd.DataFrame(x_text)
y_text_pd = pd.DataFrame(y_text)
#检查用pandas读取的数据是否是正确的
print('x_train_pd.head(5)')
print('---------')
print('y_train_pd.head(5)')
#  -------------- 2、房价训练数据导入 ----------------------------------

#  -------------- 3、数据归一化预处理 ----------------------------------
#训练集归一化
min_max_sclaer = MinMaxScaler()
min_max_sclaer.fit(x_train_pd)
x_train = min_max_sclaer.transform(x_train_pd)
x_train = torch.Tensor(x_train)#torch.Tensor()是python类，是默认张量类型torch.FloatTensor()的别名，他会调用Tensor类的构造函数_init_,生成单精度浮点类型的张量
min_max_sclaer.fit(y_train_pd)
y_train = min_max_sclaer.transform(y_train_pd)
y_train = torch.Tensor(y_train)
#测试集归一化
min_max_sclaer.fit(x_text_pd)
x_text = min_max_sclaer.transform(x_text_pd)
x_text = torch.Tensor(x_text)
min_max_sclaer.fit(y_text_pd)
y_text = min_max_sclaer.transform(y_text_pd)
y_text = torch.Tensor(y_text)
inputs = Input(shape=(404,))
#  -------------- 3、数据归一化预处理 ----------------------------------

#  --------------  4、模型构建------------------------------------------
#/---------------------menthod1-----------------------------------/
class Model1(torch.nn.model):
    def __init__(self):
        super(Model1,self).__init__()
        self.dense1 = torch.nn.linear(inputs.shape[1],10)
        self.relu1 = torch.nn.relu()
        self.dense2 = torch.nn.linear(10,15)
        self.relu2 = torch.nn.relu()
        self.dense3 = torch.nn.linear(15, 1)
    def forward(self,x):
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        return x
#/---------------------menthod1------------------------------------/
#/---------------------menthod2------------------------------------/
class Model2(torch.nn.model):
    def __init__(self):
        super(Model2,self).__init__()
        self.dense = torch.nn.Sequential(torch.nn.linear(inputs.shape[1],10),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.linear(10,15),
                                         torch.nn.linear(15,1)
                                         )
    def forward(self,x):
        x = self.dense(x)
        return x
#/---------------------menthod2------------------------------------/
#/---------------------menthod3------------------------------------/
class Model3(torch.nn.model):
    def __init__(self):
        super(Model3,self).__init()
        self.dense = torch.nn.Sequential()
        self.dense.add.module('dense1',torch.nn.linear(inputs.shape[1],10)),
        self.dense.ass.module('dropout', torch.nn.Dropout(0.2)),
        self.dense.ass.module('dense2',torch.nn.linear(10,15)),
        self.dense.ass.module('dense3', torch.nn.linear(15, 1))
    def forward(self,x):
        x = self.dense(x)
        return x
#/---------------------menthod3------------------------------------/
#/---------------------menthod4------------------------------------/
from collections import OrderedDict
class Model4(torch.nn.model):
    def __init__(self):
        super(Model4,self).__init()
        self.dense = torch.nn.Sequential(
            OrderedDict([('dense1',torch.nn.linear(inputs.shape[1],10)),
                         ('relu',torch.relu),
                         ('dense2',torch.nn.linear(10,15)),
                         ('relu', torch.relu),
                         ('dense3',torch.nn.linear(15,1))
                       ])
                                     )
    def forward(self,x):
        x = self.dense(x)
        return x
#/---------------------menthod4------------------------------------/
#  --------------  4、模型构建------------------------------------------

#  --------------  5、模型训练和保存------------------------------------
model = Model1()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)#使用Adam优化器来优化，学习率为1e-4
loss_fn = torch.nn.MSELoss()#均方误差作为巡视函数
Epoch = 200#迭代次数是200次
#前向传播
y_p = Model1(x_train)
loss = loss_fn(y_p,y_train)#计算损失函数
print(loss.item())
optimizer.zero.grad()#优化参数为0
#反向传播
loss.backward()
optimizer.step()#更新梯度
plt.plot(loss.item())
plt.show()
#保存模型
torch.save(Model1,'mlp.h6')
Model1_= torch.load('mlp.h6')
#  --------------  5、模型训练和保存-------------------------------------




