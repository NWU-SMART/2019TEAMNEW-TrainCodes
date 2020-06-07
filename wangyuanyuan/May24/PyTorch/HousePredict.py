#--------------         开发者信息--------------------------
#开发者：王园园
#开发日期：2020.5.23
#software：pycharm
#项目名称：房价预测（PyTorch）

#--------------------------导入包--------------------------
from tkinter import Variable

import torch
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from catalyst.contrib.nn import Flatten
from prompt_toolkit.input import Input
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#-------------------------加载数据--------------------------
from sklearn.tests.test_multioutput import classes
from torch import nn, optim
path = 'D:\\keras_datasets\\boston_housing.npz'  #数据地址
f = np.load(path)
#404训练数据
x_train = f['x'][:404]   #训练数据0-404
y_train = f['y'][:404]   #训练标签0-404
x_valid = f['x'][404:]   #验证数据405-505
y_valid = f['y'][404:]   #验证标签
f.close()

#--------------------------数据处理---------------------------
#将数据转成DataFrame格式
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  #取测试数据的前5个
print(y_train_pd.head(5))  #取测试标签的前5个

#用MinMaxScaler()将数据归一化，归一化到[0,1]
#训练数据归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
#训练标签归一化
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)
#验证数据归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)
#验证标签归一化
min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

#------------------------模型Method 1-------------------------------------------------
input1 = Input(shape=(404,))
class model1(torch.nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.dense1 = torch.nn.Linear(input1.shape[1], 10)
        self.dense2 = torch.nn.Linear(10, 15)
        self.dense3 = torch.nn.Linear(15, 1)
        self.dp = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dp(x)
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x

#-------------------------模型Method 2----------------------------------------------------
class model2(torch.nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.dense = torch.nn.Sequential(torch.nn.Linear(input1.shape[1], 10),
                                          torch.nn.Dropout(0.2),
                                          torch.nn.Linear(10, 15),
                                          torch.nn.Linear(15, 1)
                                          )

    def forward(self, x):
        x = self.dense(x)
        return x

#--------------------------模型Method 3---------------------------------------------------
class model3(torch.nn.Module):
    def __init__(self):
        super(model3, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('dense1', torch.nn.Linear(input1.shape[1], 10))
        self.conv.add_module('dropout', torch.nn.Dropout(0.2))
        self.conv.add_module('dense2', torch.nn.Linear(10, 15))
        self.conv.add_module('dense3', torch.nn.Linear(15, 1))

    def forward(self, x):
        x = self.conv(x)
        return x

#----------------------------模型Method4---------------------------------------------------------
class model4(torch.nn.Module):
    def __init__(self):
        super(model4, self).__init__()
        self.dense = torch.nn.Sequential(
            OrderedDict([('dense1', torch.nn.Linear(input1.shape[1], 10)),
                         ('dropout', torch.nn.Dropout(0.2)),
                         ('dense2', torch.nn.Linear(10, 15)),
                         ('dense3', torch.nn.Linear(15, 1))])
                                         )
    def forward(self, x):
        x = self.dense(x)
        return x

#--------------------------------------测试函数--------------------------------------------------
def trainandsave():
    # 神经网络结构
    model = model4()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 学习率为0.001
    criterion = nn.CrossEntropyLoss()  # 损失函数也可以自己定义，我们这里用的交叉熵损失函数
    # 训练部分
    for epoch in range(5):  # 训练的数据量为5个epoch，每个epoch为一个循环
        # 每个epoch要训练所有的图片，每训练完成200张便打印一下训练的效果（loss值）
        running_loss = 0.0  # 定义一个变量方便我们对loss进行输出
        for i in enumerate(x_train, y_train, 0):  # 这里我们遇到了第一步中出现的训练数据，代码传入数据
            # enumerate是python的内置函数，既获得索引也获得数据
            inputs, labels = Variable(x_train), Variable(y_train)  # 转换数据格式用Variable

            optimizer.zero_grad()  # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度

            # forward + backward + optimize
            outputs = model(inputs)  # 把数据输进CNN网络net
            loss = criterion(outputs, labels)  # 计算损失值
            loss.backward()  # loss反向传播
            optimizer.step()  # 反向传播后参数更新
            running_loss += loss.data[0]  # loss累加
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))  # 然后再除以200，就得到这两百次的平均损失值
                running_loss = 0.0  # 这一个200次结束后，就把running_loss归零，下一个200次继续使用

    print('Finished Training')
    # 保存神经网络
    torch.save(model, 'net.pkl')  # 保存整个神经网络的结构和模型参数
    torch.save(model.state_dict(), 'net_params.pkl')  # 只保存神经网络的模型参数

#调用保存的模型
def reload_net():
    trainednet = torch.load('net.pkl')
    return trainednet

#对验证数据的处理。格式转换，不转换，数据无法正常读取
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#-------------------------------------------测试函数------------------------------------
def test():
    model = reload_net()
    imshow(torchvision.utils.make_grid(x_valid, nrow=5))  # nrow是每行显示的图片数量，缺省值为8
    print('GroundTruth: '
          , " ".join('%5s' % classes[y_valid[j]] for j in range(25)))  # 打印前25个GT（test集里图片的标签）
    outputs = model(Variable(x_train))
    _, predicted = torch.max(outputs.data, 1)

    print('Predicted: ', " ".join('%5s' % classes[predicted[j]] for j in range(25)))
    # 打印前25个预测值

trainandsave()  #调用训练函数
test()          #调用测试函数
