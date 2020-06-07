# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年5月22日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import os
import functools
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
from .Flatten import *
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、数据载入与预处理 -------------------------------
def load_data():
    paths = [
        'D:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'D:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]
    # 加载数据返回4个NumPy数组
    with gzip.open(paths[0], 'rb') as lbpath: # 读压缩文件
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        # frombuffer将data以流的形式读入转化成ndarray对象
        # 第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()# 载入数据
x_train = x_train.astype('float32')#数据类型转换
x_test = x_test.astype('float32')
x_train /= 255  #归一化
x_test /= 255
y_train = LabelBinarizer().fit_transform(y_train)# 对y进行one-hot编码
y_test = LabelBinarizer().fit_transform(y_test)
y_train =np.array(y_train)
y_test =np.array(y_test)
x_train = Variable(torch.from_numpy(x_train))# 变为variable数据类型
y_train = Variable(torch.from_numpy(y_train))
x_test = Variable(torch.from_numpy(x_test))
y_test = Variable(torch.from_numpy(y_test))
#  -------------------------- 2、数据载入与预处理 -------------------------------

#  -------------------------- 4、模型训练以及保存   --------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1), #输入通道1，输出通道32，3x3的卷积核，步长1，padding 1
            torch.nn.ReLU() #relu层
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 3, 1, 0),#输入通道32，输出通道32，3x3的卷积核，步长1，padding 0
            torch.nn.ReLU(),#relu层
            torch.nn.MaxPool2d(2)#最大池化层
        )
        self.dropout1 = torch.nn.Dropout(0.25) # dropout设置为0.25
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),#输入通道32，输出通道64，3x3的卷积核，步长1，padding 1
            torch.nn.ReLU()#relu层
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 0),#输入通道64，输出通道64，3x3的卷积核，步长1，padding 0
            torch.nn.ReLU(),#relu层
            torch.nn.MaxPool2d(2)#最大池化层
        )
        self.dropout2 = torch.nn.Dropout(0.25)# dropout设置为0.25
        self.linear1 = torch.nn.Flatten()#平展
        self.linear2 = torch.nn.Linear(1600, 512)# 全连接层
        self.linear3 = torch.nn.ReLU()#relu层
        self.linear4 = torch.nn.Dropout(0.5)# dropout设置为0.5
        self.linear5 = torch.nn.Linear(512, 10) # 全连接层分10类
        self.linear6 = torch.nn.Softmax()#softmax激活函数
    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.dropout2(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        out = self.linear6(x)
        return out
model = Net()
learning_rate = 1e-4 # 学习率
EPOCH = 5  # epoch,迭代多少次
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for i in range(EPOCH):
    # 向前传播
    y_pred= model(x_train)
    # 计算损失
    loss = loss_func(y_pred, y_train)
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新梯度
    optimizer.step()

    if (i+1) % 1 == 0:#每训练1个epoch，打印一次损失函数的值
        print(loss.data)
    if (i + 1) % 5 == 0: #每训练5个epoch,保存一次模型
        torch.save(model.state_dict(), "./model.pkl")  # 保存模型
        print("save model")
#  -------------------------- 4、模型训练以及保存   --------------------------------

