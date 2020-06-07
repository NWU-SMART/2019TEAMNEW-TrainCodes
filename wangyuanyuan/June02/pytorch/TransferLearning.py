#------------------------------------------------开发者信息-------------------------------------------
#开发人：王园园
#开发日期：2020.6.2
#开发软件：pycharm
#开发项目：图像分类：迁移学习（pytorch）

#---------------------------------------------------导包----------------------------------------------
import gzip
import cv2
import numpy as np
import torch
from sklearn.preprocessing import LabelBinarizer
from torch import nn
from torch.autograd import Variable
from torchxrayvision import models

#--------------------------------------------------读取数据及图像预处理----------------------------------
path = 'D:/keras_datasets/'
def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]
    #训练数据标签
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #训练数据
    with gzip.open(path[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    #测试数据标签
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #测试数据
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()

#对标签进行独热编码
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

#由于mist的输入数据维度是（num， 28， 28）， vgg16需要三维图像，因为扩充一下mnist的最后一维
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_DRAY2RGB) for i in x_test]
#数组
x_train = np.asarray(X_train)
x_test = np.asarray(x_test)
#数据类型是float32
x_train = x_train.astype('float32')
x_test = x_test.astype(('float32'))
#数据归一化
x_train /= 255
x_test /= 255

#变为variable数据类型，类似于keras里的tensor，但比tensor有更多的属性
x_train = Variable(torch.from_numpy(x_train))
y_train = Variable(torch.from_numpy(y_train))
x_test = Variable(torch.from_numpy(x_test))
y_test = Variable(torch.from_numpy(y_test))

#--------------------------------------------------------构建模型--------------------------------------
model = models.vgg16(pretrained=True)
#冻结vgg16的模型参数，使之不再改变
for parma in model.parameters():
    parma.requires_grad = False

model.classifier = torch.nn.Sequential(nn.Linear(7*7*512, 256),
                                       nn.Softmax(True),
                                       nn.Linear(256, 10),
                                       nn.Softmax(True))

#---------------------------------------------------------训练模型--------------------------------------
loss_function = nn.CrossEntropyLoss()  #定义损失函数
optimizer = nn.optim.Adam(model.classifier.parameters(), lr=0.00001)  #优化器
for epoch in range(5):
    output = model(x_train)    #输入训练数据，获取输出
    loss = loss_function(output, y_train)   #输出和训练数据计算损失函数
    optimizer.zero_grad()      #梯度清零
    loss.backward()            #反向传播
    optimizer.step()           #梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, 5, loss.item())) ##每训练1个epoch，打印一次损失函数的值