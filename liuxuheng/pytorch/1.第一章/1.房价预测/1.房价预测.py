# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年5月19日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
# 数据存放路径
path = 'D:\\keras_datasets\\boston_housing.npz'
# 加载数据
f = np.load(path)
# 404个训练，102个测试
x_train=f['x'][:404]  # 下标0到下标403
y_train=f['y'][:404]
x_valid=f['x'][404:]  # 下标404到最后
y_valid=f['y'][404:]
f.close()
# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
#  -------------------------- 2、房价训练和测试数据载入 -------------------------------

#  -------------------------- 3、数据归一化 -------------------------------
# 训练集归一化
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
#  -------------------------- 3、数据归一化  -------------------------------

#  -------------------------- 4、模型训练以及保存   --------------------------------
class ThreeLayerNet(nn.Module):
    def __init__(self,input_size):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, 10) # 第一层,输入大小为input_size,输出大小为10
        self.relu1 = torch.nn.ReLU()# relu激活函数
        self.linear2 = torch.nn.Linear(10, 15)  # 第二层，输入大小为10，输出大小为15
        self.relu2 = torch.nn.ReLU()# relu激活函数
        self.linear3 = torch.nn.Linear(15, 1)# 第三层，输入大小为15，输出大小为1
    def forward(self, x): # 在forward函数中，接受数据类型为Variable，返回数据类型也是Varible
        x = self.linear1(x)# 输入x经过第一层
        x = self.relu1(x) # 经过激活函数
        x = self.linear2(x) # 经过第二层
        x = self.relu2(x) # 经过激活函数
        y_pred = self.linear3(x) # 经过第三层
        return y_pred # 返回输出结果
input_size = x_train_pd.shape[1] # 输入大小也就是列的大小


x = Variable(torch.from_numpy(x_train)).float() # x_train变为variable数据类型
y = Variable(torch.from_numpy(y_train)).float() # y_train变为variable数据类型
x_val = Variable(torch.from_numpy(x_valid)).float() # x_valid变为variable数据类型
y_val = Variable(torch.from_numpy(y_valid)).float() # y_valid变为variable数据类型
model = ThreeLayerNet(input_size) # 定义model
loss_fn = nn.MSELoss(reduction='sum') #损失函数，均方误差
learning_rate = 1e-4 # 学习率
EPOCH = 10000  # epoch,迭代多少次
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #SGD优化器
for i in range(EPOCH):
    # 向前传播
    y_pred= model(x)
    # 计算损失
    loss = loss_fn(y_pred, y)
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新梯度
    optimizer.step()

    if (i+1) % 100 == 0:#每训练100个epoch，打印一次损失函数的值
        print(loss.data)
    if (i + 1) % 500 == 0: #每训练500个epoch,保存一次模型
        torch.save(model.state_dict(), "./model.pkl")  # 保存模型
        print("save model") 
#  -------------------------- 4、模型训练以及保存    -------------------------------

#  -------------------------- 5、加载模型并预测    ------------------------------
model.load_state_dict(torch.load("./model.pkl",map_location=lambda storage, loc: storage)) # 加载训练模型
print("load model")
y_val_pred =model (x_val)#验证集验证，输入x_val，获得y_val的预测值 
print(y_val_pred) #输出y_val的预测值
yyy = y_val_pred.detach().numpy() # 因为之前对数值进行了归一化，所以这里需要反归一化，需要将y_val_pred变为numpy数组形式，但是由于存在梯度，无法直接变，需要加上detach()
y_yuce = min_max_scaler.inverse_transform(yyy) # 进行反归一化
print(y_yuce) # 输出反归一化后的预测值
loss =loss_fn(y_val_pred,y_val) # 计算预测值和真实的验证数据的损失函数
print(loss)#打印损失函数
#  -------------------------- 5、加载模型并预测    ------------------------------


