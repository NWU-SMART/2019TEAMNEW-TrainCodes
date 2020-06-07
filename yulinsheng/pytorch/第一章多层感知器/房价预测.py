#------------------ 开发者信息 --------------------*/
'''
      开发者：于林生
 *    开发日期：2020.5.20
 *    版本号：Versoin 1.0
 *    修改日期：
 *    修改人：
 *    修改内容：
 *
'''
# /*------------------ 开发者信息 --------------------*/

# /*------------------ 代码布局 --------------------*/
'''
代码布局
1.导入需要的包
2.读取数据
3.数据预处理
4.模型构建
5.训练模型
6.结果可视化
'''

# /*------------------ 代码布局 --------------------*/


# /*------------------ 导入需要的包 --------------------*/
# 导入需要的包文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# /*------------------ 导入需要的包 --------------------*/

# /*------------------ 读取数据 --------------------*/
# 读取数据
path = "G:/python/code/多层感知器/boston_housing.npz"  #数据的路径
data = np.load(path)#加载npz数据
print(data.files)#打印数据包含的表头
# 获得数据，并查看数据类型
y = data['y']
x = data['x']
print(y.shape)
print(x.shape)

# 划分数据集（训练集和验证集）
train_x = x[:404]
test_x = x[404:]
train_y = y[:404]
test_y = y [404:]
data.close()
print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)

# /*------------------ 读取数据 --------------------*/

# /*------------------ 数据预处理 --------------------*/

# 数据归一化
# 转成DataFrame格式方便数据处理(矩阵类型)
x_train_pd = pd.DataFrame(train_x)#（404*13）
y_train_pd = pd.DataFrame(train_y)#（404*1）
x_valid_pd = pd.DataFrame(test_x)#（102*13）
y_valid_pd = pd.DataFrame(test_y)#（102*1）
# 训练集归一化归一到0-1之间
min_max_scaler = MinMaxScaler()

min_max_scaler.fit(x_train_pd)
train_x = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
train_y = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
test_x = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
test_y = min_max_scaler.transform(y_valid_pd)
# 上述数据处理结果为numpy类型
# 转换为tensor类型
train_x = torch.FloatTensor(train_x)
train_y = torch.FloatTensor(train_x)
test_x = torch.FloatTensor(train_x)
test_y = torch.FloatTensor(train_x)


# /*------------------ 数据预处理 --------------------*/


# /*------------------ 模型构建 --------------------*/
# 模型构建
# 三层网络结果
# /*------------------ 第一种sequential --------------------*/
# model = torch.nn.Sequential(
# #     输入特征为X的13维特征，输出设定为100
#     torch.nn.Linear(13, out_features=100),
# #     通过一个relu激活函数
#     torch.nn.ReLU(),
# #     根据上一个输入100，定义为一个输出
#     torch.nn.Linear(100, 1),
# #     再跟随一个relu激活函数
# #     torch.nn.ReLU(),
# )
# /*------------------ 第二种sequential + OrderedDict类型 --------------------*/
from collections import OrderedDict
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.model = torch.nn.Sequential(
            OrderedDict([
                ('dense1',torch.nn.Linear(13,10)),
                ('relu',torch.nn.ReLU()),
                ('dense2',torch.nn.Linear(10,15)),
                ('relu',torch.nn.ReLU()),
                ('dense3',torch.nn.Linear(15,1))
            ]))
    def forward(self,x):
        result = self.model(x)
        return result

# /*------------------ 第三种利用类继承add_module的方式 --------------------*/
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.input = torch.nn.Sequential()
#         self.input.add_module('dense1',torch.nn.Linear(13,10))
#         self.input.add_module('relu1',torch.nn.ReLU())
#         self.input.add_module('dense2',torch.nn.Linear(10,15))
#         self.input.add_module('relu2', torch.nn.ReLU())
#         self.input.add_module('dense3',torch.nn.Linear(15,1))
#     def forward(self,x):
#         result = self.input(x)
#         return result



# /*------------------ 第四种利用类继承方式 --------------------*/
# class Model(torch.nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.dense1 = torch.nn.Linear(13,10)
#         self.relu = torch.nn.ReLU()
#         self.dense2 = torch.nn.Linear(in_features=10,out_features=15)
#         self.dense3 = torch.nn.Linear(in_features=15, out_features=1)
#     def forward(self,x):
#         x = self.dense1(x)
#         x = self.relu(x)
#         x = self.dense2(x)
#         x = self.relu(x)
#         result = self.dense3(x)
#         return result


#

model = Model()
# 定义学习率
learning_rate = 1e-3
# 定义损失函数MSE
loss_fn = torch.nn.MSELoss(reduction='sum')
# 优化函数Adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# /*------------------ 模型构建 --------------------*/

# /*------------------ 模型训练--------------------*/
# 初始化参数
# torch.nn.init.normal_(model[0].weight)
# torch.nn.init.normal_(model[2].weight)

num_epochs = 200
print(train_x.shape)
for i in range(num_epochs):
#     训练特征
    y_pred = model(train_x)
# 计算损失函数
    loss = loss_fn(y_pred,train_y)
    print(i, loss.item())
# 优化参数归零
    optimizer.zero_grad()
#     反向传播
    loss.backward()
    optimizer.step()
    plt.plot(i,loss.item())
    plt.scatter(i,loss.item())
plt.show()

# /*------------------ 模型训练--------------------*/
# 保存模型
torch.save(model,'mlp.pkl')
model_1 = torch.load('mlp.pkl')
model_1.eval()
y_predtest = model(test_x)
loss1= loss_fn(y_predtest,test_y)
print(loss)