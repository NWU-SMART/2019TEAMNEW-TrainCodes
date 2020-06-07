# ----------------开发者信息--------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年5月27日
# 开发框架：pytorch
#-----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 torch, matplotlib, numpy, sklearn 和 panda 的包
# 2、参数定义
# 3、房价训练和测试数据载入
# 4、数据归一化
# 5、模型训练
# 6、模型测试
# 7、模型可视化
#--------------------------------------------------------------#

#  -------------------------- 1、导入需要包 -------------------------------
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  # 导入包
#  -------------------------- 导入需要包 -------------------------------

#  -------------------------- 2、参数定义 -------------------------------
EPOCH = 200  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
LR = 1e-2  # 学习率
#  -------------------------- 参数定义 -------------------------------

#  -------------------------- 3、房价训练和测试数据载入 -------------------------------
# 数据放到本地路径
path = 'E:\\study\\kedata\\boston_housing.npz'
f = np.load(path)
# 404个训练数据
x_train = f['x'][:404]   #训练数据下标0-403
y_train = f['y'][:404]   #训练标签下标0-403
# 102个验证数据
x_valid = f['x'][404:]   #验证数据下标404-505
y_valid = f['y'][404:]   #验证标签下标404-505
f.close()
# 数据放到本地路径

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(3))  # 输出训练数据的x (前3个)
print(y_train_pd.head(3))  # 输出训练数据的y (前3个)
# 转成DataFrame格式方便数据处理
#  -------------------------- 房价训练和测试数据载入 -------------------------------

#  -------------------------- 4、数据归一化 -------------------------------
# 归一化函数
min_max_scaler = MinMaxScaler()  # 归一到 [ 0，1 ]函数
# 训练集归一化
min_max_scaler.fit(x_train_pd) # 计算最大值最小值
x_train = min_max_scaler.transform(x_train_pd) # 归一化
x_train = torch.Tensor(x_train) # 转化为tensor形式
x_train = x_train.float()  # double转变为float

min_max_scaler.fit(y_train_pd) # 计算最大值最小值
y_train = min_max_scaler.transform(y_train_pd) # 归一化
y_train = torch.Tensor(y_train)  # 转化为tensor形式
y_train = y_train.float()  # double转变为float

# 测试集归一化
min_max_scaler.fit(x_valid_pd) # 计算最大值最小值
x_valid = min_max_scaler.transform(x_valid_pd) # 归一化
x_valid = torch.Tensor(x_valid) # 转化为tensor形式
x_valid = x_valid.float()  # double转变为float

min_max_scaler.fit(y_valid_pd) # 计算最大值最小值
y_valid = min_max_scaler.transform(y_valid_pd) # 归一化
y_valid = torch.Tensor(y_valid) # 转化为tensor形式
y_valid = y_valid.float()  # double转变为float
#  -------------------------- 数据归一化 -------------------------------

#  -------------------------- 5、模型构造  -------------------------------
input_size = x_train_pd.shape[1]
class MLP_model(nn.Module):
    def __init__(self):
        super(MLP_model, self).__init__()
        self.layer1 = nn.Linear(input_size,10)
        self.relu1 = nn.ReLU()
        self.Dropout = torch.nn.Dropout(0.2)
        self.layer2 = nn.Linear(10,15)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(15,1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.Dropout(x)
        x = self.layer2(x)
        x = self.relu2(x)
        out = self.layer3(x)
        return out

model = MLP_model()
print(model)
#  -------------------------- 5、模型训练  -------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.MSELoss(reduction='sum')
print("-----------训练开始-----------")
iteration = []  # list存放epoch数
loss_total = []  # list存放损失
for epoch in range(EPOCH):
        # train_loss = 0.0
        model.train()  # 训练模式
        predict = model(x_train)  # output
        loss_epoch_train = loss_func(y_train,predict)  # cross entropy loss
        iteration.append(epoch)  # 将epoch放到list中
        loss_total.append(loss_epoch_train)  # 将loss放到list中
        optimizer.zero_grad()  # clear gradients for this training step
        loss_epoch_train.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        print('epoch %3d , loss %3d' % (epoch, loss_epoch_train))
print("-----------训练结束-----------")
torch.save(model.state_dict(), "housetorch.pkl")  # 保存模型参数
# -------------------------------模型训练------------------------

#  -------------------------- 6、模型测试  -------------------------------
print("-----------测试开始-----------")
model.load_state_dict(torch.load('housetorch.pkl'))  # 加载模型
iteration_test = []  # list存放epoch数
loss_total_test = []  # list存放损失
for epochs in range(EPOCH):
        model.eval()  # 测试模式
        predict_test = model(x_valid)  # output
        loss_epoch_test = loss_func(y_valid,predict_test)  # cross entropy loss
        iteration_test.append(epochs)  # 将epoch放到list中
        loss_total_test.append(loss_epoch_test)  # 将loss放到list中
        print('epoch %3d , loss %3d' % (epochs, loss_epoch_test))
print("-----------测试结束-----------")
#  -------------------------- 模型测试  -------------------------------

#  -------------------------- 7、模型可视化    ------------------------------
plt.plot(iteration,loss_total, label="Train loss")
plt.plot(iteration_test, loss_total_test, label="Test loss")  # iteration和loss对应画出
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left') # loc:图例位置
plt.show()
#  -------------------------- 模型可视化    ------------------------------

