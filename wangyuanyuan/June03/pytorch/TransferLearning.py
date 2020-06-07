#---------------------------------------------------开发者信息----------------------------------------------
#开发人：王园园
#开发日期：2020.6.03
#开发软件：pycharm
#开发项目：图像回归：迁移学习（pytorch）

#---------------------------------------------------导包---------------------------------------------------
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
from torchxrayvision import models


#------------------------------------------------------读取手写提数据及图像预处理-----------------------------
path = 'D:\keras_datasets\mnist.npz'
f = np.load(path)
x_train = f['x_train']    #训练数据
y_train = f['y_train']    #训练数据标签
x_test = f['x_test']       #测试数据
y_test = f['y_test']       #测试数据标签
f.close()
#数据标准化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#np.prod是将28*28矩阵转化成1*784，方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

#伪造回归数据
#转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
#设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']
#给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 33]  #均值列表
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]  #均值mean，标准差std，数量
    return np.round(price, 2)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)
#数据归一化
#训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train_label = min_max_scaler.transform(y_train_pd)[:, 1]
#验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]

#变为variable数据类型，类似于keras里的tensor，但比tensor有更多的属性
y_train = Variable(torch.from_numpy(y_train_label))
y_test = Variable(torch.from_numpy(y_test_label))

#--------------------------------------------------------构建模型-------------------------------------
model = models.VGG16(pretrained=True)
for layer in model.layers[:15]:
    layer.trainable = False
model.classifier = torch.nn.Sequential(nn.Linear(28*28, 256),
                                       nn.ReLU(True),
                                       nn.Dropout(0.5),
                                       nn.Linear(256, 1),
                                       nn.Softmax(True))
loss_function = nn.CrossEntropyLoss()  #定义损失函数
optimizer = nn.optim.Adam(model.classifier.parameters(), lr=0.00001)  #优化器
for epoch in range(5):
    output = model(x_train)    #输入训练数据，获取输出
    loss = loss_function(output, y_train)   #输出和训练数据计算损失函数
    optimizer.zero_grad()      #梯度清零
    loss.backward()            #反向传播
    optimizer.step()           #梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, 5, loss.item())) ##每训练1个epoch，打印一次损失函数的值
