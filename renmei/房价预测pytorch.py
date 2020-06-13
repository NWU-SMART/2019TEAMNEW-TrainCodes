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
from sklearn.preprocessing import MinMaxScaler

from torchvision import transforms
#  -------------------------- 1、导入需要包 -------------------------------



#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
# 数据在服务器可以访问
# train_data.shape:(404, 13),test_data.shape:(102, 13),
# train_targets.shape:(404,),test_targets.shape:(102,)
# the data compromises 13 features
# (x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()  # 加载数据（国外服务器无法访问）

# 数据放到本地路径
# D:\\keras_datasets\\boston_housing.npz(本地路径)
path = 'C:\\Users\\Administrator\\Desktop\\代码\\boston_housing.npz'
f=np.load(path)
# 404个训练，102个测试
# 训练数据
x_train = f['x'][:404]  # 下标0到下标403
y_train = f['y'][:404]
# 测试数据
x_valid = f['x'][404:]  # 下标404到下标505
y_valid = f['y'][404:]
f.close()
#x_train=torch.autograd.Variable(torch.from_numpy(x_train)).float()
#y_train=torch.autograd.Variable(torch.from_numpy(y_train)).float()
#x_valid=torch.autograd.Variable(torch.from_numpy(x_valid)).float()
#y_train=torch.autograd.Variable(torch.from_numpy(y_valid)).float()

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
x_train=torch.Tensor(x_train).float()
y_train=torch.Tensor(y_train).float()
x_valid=torch.Tensor(x_valid).float()
y_valid=torch.Tensor(y_valid).float()


#  -------------------------- 4、模型训练

epoch=5
lr=0.03
"""
net=nn.Sequential(
    nn.Linear(x_train_pd.shape[1],10),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(10,15),
    nn.ReLU(),
    nn.Linear(10,1),

)
"""
class house(nn.Module):
    def __init__(self):
        super(house,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(x_train.shape[1],10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 1),

        )
    def forward(self,x):
        out=self.fc(x)
        return out
net=house()


loss=nn.MSELoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr)
train_losses=[]
test_losses=[]

for i in range(epoch):

    train_loss=0
    net.train()
    out=net(x_train)
    epoch_loss=loss(out,y_train)
    optimizer.zero_grad()
    epoch_loss.backward()
    optimizer.step()
    train_losses.append(epoch_loss/len(x_train))
    test_loss=0
    net.eval()
    y=net(x_valid)
    eval_loss=loss(y,y_valid)
    test_losses.append(eval_loss/len(x_valid))
    print("第%d次迭代,训练集损失为%f,验证集损失为为%f"%(i+1,epoch_loss,eval_loss))
predict=net(x_valid)
predict_numpy = predict.detach().numpy()
min_max_scaler.fit(y_valid_pd)
predict = min_max_scaler.inverse_transform(predict_numpy)  # 反归一化
print(predict)
plt.plot(train_losses,)
plt.plot(test_losses)
plt.title('torch loss')  #
plt.xlabel('Epoch')  #
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

