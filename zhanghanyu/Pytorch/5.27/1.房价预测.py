# ----------------开发者信息----------------------------
# 开发者：张涵毓
# 开发日期：2020年5月27日
# 内容：房价预测pytorch
# 修改时间：2020年6月1日
# 修改人：张涵毓
# 修改内容：用pytorch的4种方法构建房价预测的网络模型
# ----------------开发者信息----------------------------

# ----------------------代码布局-------------------------------------
# 1.引入pytorch，numpy，sklearn，pandas包
# 2.导入数据
# 3.数据归一化
# 4.模型建立
# 5.加载模型并预测

# ---------------------------------------------------------------------

#--------1、导入需要包----------#
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn as nn
import torch
from torch.autograd import Variable
#--------1、导入需要包----------#

#--------2、导入数据-----------#
path='D:\\研究生\\代码\\Keras代码\\1.Multi-Layer perceptron(MLP 多层感知器)\\boston_housing.npz' #数据路径
data = np.load(path)  # 读取路径数据
#404个训练 102个验证
#训练数据
train_x = data['x'][0:404]  # 前404个数据为训练集 0-403
train_y = data['y'][0:404]
#验证数据
valid_x = data['x'][404:]  # 后面的102个数据为测试集 404-505
valid_y = data['y'][404:]
data.close()

# 转成DataFrame格式方便数据处理
train_x_pd = pd.DataFrame(train_x)
train_y_pd = pd.DataFrame(train_y)
valid_x_pd = pd.DataFrame(valid_x)
valid_y_pd = pd.DataFrame(valid_y)

# 查看训练集前五条数据
print(train_x_pd.head(5))
print(train_y_pd.head(5))
#--------2、导入数据-----------#

#  ------------ 3、数据归一化 ------------#
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(train_x_pd)
train_x = min_max_scaler.transform(train_x_pd)

min_max_scaler.fit(train_y_pd)
train_y = min_max_scaler.transform(train_y_pd)

# 验证集归一化
min_max_scaler.fit(valid_x_pd)
valid_x = min_max_scaler.transform(valid_x_pd)

min_max_scaler.fit(valid_y_pd)
valid_y = min_max_scaler.transform(valid_y_pd)
#  ------------ 3、数据归一化 ------------#

#  -------------4、模型建立训练保存--------------#
#  --------------Method-1:早期最常用的方法------------------#
class MLPhp(nn.Module):
    def __init__(self,input_size):
        super(MLPhp,self).__init__()
        # 第一层,输入大小为input_size,输出大小为10,relu激活函数
        self.linear1 = torch.nn.Linear(input_size, 10)
        self.relu1 = torch.nn.ReLU()
        # 第二层，输入大小为10，输出大小为15,relu激活函数
        self.linear2 = torch.nn.Linear(10, 15)
        self.relu2 = torch.nn.ReLU()
        # 第三层，输入大小为15，输出大小为1,最后一层不用激活函数
        self.linear3 = torch.nn.Linear(15, 1)

    #前向传递，输出结果
    def forward(self, x):
            x = self.linear1(x)
            x = self.relu1(x)
            x = self.linear2(x)
            x = self.relu2(x)
            y_pred = self.linear3(x)
            return y_pred
#  --------------Method-1:早期最常用的方法------------------#
#  --------------Method-2:利用torch.nn.Sequential()容器进行快速搭建------------------#
'''
class MLPhp(nn.module):
    def __init__(self,input_size):
        super(MLPhp, self).__init__()
        self.dense = nn.Sequential(nn.linear(input_size, 10),
                                   nn.ReLU(),
                                   nn.linear(10, 15),
                                   nn.ReLU(),
                                   nn.linear(15, 1)
                                         )

    def forward(self, x):
        y_pred = self.dense(x)
        return y_pred
'''
#  ---Method-2:利用torch.nn.Sequential()容器进行快速搭建，但缺点是每层命名为默认阿拉伯数字，不易区分------------------#
#  -------------Method-3：对方法2的改进，通过add.module()添加每一层------------------#
'''
class MLPhp(nn.model):
    def __init__(self):
        super(MLPhp,self).__init()
        self.dense = nn.Sequential()
        self.dense.add.module('dense1', nn.linear(input_size,10)),
        self.dense.add.module('relu1', nn.ReLU()),
        self.dense.add.module('dense2', nn.linear(10,15)),
        self.dense.add.module('relu2', nn.ReLU()),
        self.dense.add.module('dense3', nn.linear(15, 1))
    def forward(self,x):
        y_pred = self.dense(x)
        return y_pred
'''
#  ---------Method-3：对方法2的改进，通过add.module()添加每一层，每一层可以有单独的名字------------------#
#  -------------------------Method-4:方法3的另一种写法---------------------------#
'''
from collections import OrderedDict
class MLPhp(nn.module):
    def __init__(self,input_size):
        super(MLPhp,self).__init__()
        self.dense=nn.Sequential(
            OrderedDict([
                    ("dense1",nn.linear(input_size,10)),
                    ("relu1",nn.ReLU()),
                    ("dense2",nn.linear(10,15)),
                    ("relu2",nn.ReLU()),
                    ("dense3",nn.linear(15,1))

                ])
        )
    def forward(self,x):
        y_pred=self.dense(x)
        return y_pred
'''
#  -------------------------Method-4:方法3的另一种写法---------------------------#

input_size = train_x_pd.shape[1]  # 输入大小也就是列的大小
    # 变为variable数据类型
x = Variable(torch.from_numpy(train_x)).float()
y = Variable(torch.from_numpy(train_y)).float()
x_val = Variable(torch.from_numpy(valid_x)).float()
y_val = Variable(torch.from_numpy(valid_y)).float()
    # 定义model
model = MLPhp(input_size)
loss_fn = nn.MSELoss(reduction='sum')  # 损失函数，均方误差
learning_rate = 1e-4  # 学习率
EPOCH = 10000  # 迭代次数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # SGD优化

for i in range(EPOCH):
        # 向前传播
    y_pred = model(x)

    loss = loss_fn(y_pred, y)# 计算损失

    optimizer.zero_grad() # 梯度清零

    loss.backward() # 反向传播

    optimizer.step() # 更新梯度

    if (i + 1) % 100 == 0:  # 每训练100个epoch，打印一次损失函数的值
        print(loss.data)
    if (i + 1) % 500 == 0:  # 每训练500个epoch,保存一次模型
        torch.save(model.state_dict(), "./model.pkl")  # 保存模型
        print("save model")
#  -------------4、模型建立训练保存--------------#
#---------------5、加载模型并预测 - --------------#
model.load_state_dict(torch.load("./model.pkl", map_location=lambda storage, loc: storage))  # 加载训练模型
print("load model")
y_val_pred = model(x_val)  # 验证集验证，输入x_val，获得y_val的预测值
print(y_val_pred)  # 输出y_val的预测值
y_new = y_val_pred.detach().numpy()  # 因为之前对数值进行了归一化，所以这里需要反归一化，需要将y_val_pred变为numpy数组形式，但是由于存在梯度，无法直接变，需要加上detach()
y_yuce = min_max_scaler.inverse_transform(y_new)  # 进行反归一化
print(y_yuce)  # 输出反归一化后的预测值
loss = loss_fn(y_val_pred, y_val)  # 计算预测值和真实的验证数据的损失函数
print(loss)  # 打印损失函数
#---------------5、加载模型并预测 - --------------#
