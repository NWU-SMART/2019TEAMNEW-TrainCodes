# ----------------开发者信息-----------------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月5日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息------------------------------------------------
# ----------------------   代码布局： ----------------------
# 1、导入需要的的包
# 2、数据读取
# 3、数据预处理
# 4、建立模型
# 5、训练模型
# 6、保存模型及显示结果
# ----------------------   代码布局： ----------------------
#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import os
import functools
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import LabelBinarizer
from .Flatten import *
#  -------------------------- 2、读取数据---------------------------------
from tensorflow_core.python.ops.gen_state_ops import Variable


def load_data():
    path = ['D:/northwest/小组视频/4图像分类/train-labels-idx1-ubyte.gz','D:/northwest/小组视频/4图像分类/train-images-idx3-ubyte.gz'
           'D:/northwest/小组视频/4图像分类/t10k-labels-idx1-ubyte.gz','D:/northwest/小组视频/4图像分类/t10k-images-idx3-ubyte.gz'
            ]
    with gzip.open(path[0],'rb') as lpath:
        y_train = np.frombuffer(lpath.read(),np.unit8,offset=8)
    with gzip.open(path[1],'rb') as ipath:
        x_train = np.frombuffer(ipath.read(),np.unit8,offset=16).reshape(len(y_train),28,28,1)
    with gzip.open(path[2],'rb') as lpath:
        y_text = np.frombuffer(ipath.read(),np.unit8,offset=8)
    with gzip.open(path[3],'rb') as ipath:
        x_text = np.frombuffer(ipath.read(),np.unit8,offset=16).reshape(len(y_text),28,28,1)
    return(y_train,x_train),(y_text,x_text)
(y_train,x_train),(y_text,x_text) = load_data()
#  -------------------------- 2、读取数据---------------------------------
#  -------------------------- 3、数据预处理-------------------------------
x_train = x_train.astype('float32')#将图像信息转换为数据类型
x_text = y_train.astype('float32')
x_train /= 255    #归一化
x_text /=255
x_train = Variable(torch.from_numpy(x_train))  # x_train变为variable数据类型
x_test = Variable(torch.from_numpy(x_text))    # x_test变为variable数据类型
x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)#转换为tensor类型
y_test = torch.LongTensor(y_text)
#  -------------------------- 3、数据预处理-------------------------------
#  -------------------------- 4、建立模型---------------------------------
class Model(nn.Modual):
    def __init__(self):
        super(Model,self).__init()
        self.layer = torch.nn.Sequential(
            torch.nn.Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'),
            torch.nn.Conv2D(32,(3,3),activation='relu'),
            torch.nn.MaxPooling2D(pool_size=(2, 2)),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2D(64, (3, 3), padding='same', activation='relu'),
            torch.nn.Conv2D(64, (3, 3), activation='relu'),
            torch.nn.MaxPooling2D(pool_size=(2, 2)),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Dense(512,activation='relu'),
            torch.nn.Dropout(0.5),
            torch.nn.Dense(10, activation='softmax'),
        )
    def forward(self,x):
        x= self.layer(x)
        return x
#  -------------------------- 4、建立模型---------------------------------
#  -------------------------- 5、训练模型---------------------------------
model = Model()
optimizer = torch.nn.Adam(model.parameter(),lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
epoch=5
#开始训练
for i in range(epoch):
    y_p = model(x_train)
    loss = loss_fn(y_p,y_train)
    if(i+1)%1==0:
        print(loss.item)#每迭代一次，打印一次损失值item()得到元素张量里面的元素值
    if(i+1)%5==0:
        torch.save(model.state_dict(),"./model.ic")#每迭代5次保存一次模型
    print('save model')
optimizer.zero_grad()#在进行梯度更新之前，先用optimier对象提供的清除已经积累的梯度，梯度置零
loss.backward()#计算梯度，反向传播
optimizer.step()#更新梯度
#  -------------------------- 5、训练模型---------------------------------