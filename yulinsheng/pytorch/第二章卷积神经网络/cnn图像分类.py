# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.1.2 MLP-招聘信息文本分类
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# /------------------ 导入需要的包--------------------*/


import numpy as np
import gzip
import torch
# /------------------ 导入需要的包--------------------*/

# /------------------ 读取数据--------------------*/
path_trainLabel = 'train-labels-idx1-ubyte.gz'
path_trainImage = 'train-images-idx3-ubyte.gz'
path_testLabel = 't10k-labels-idx1-ubyte.gz'
path_testImage = 't10k-images-idx3-ubyte.gz'
# /------------------ 读取数据--------------------*/

# /------------------ 数据处理--------------------*/
# 将文件解压并划分为数据集
with gzip.open(path_trainLabel,'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)#通过还原成类型ndarray。
with gzip.open(path_trainImage,'rb') as Imgpath:
    x_train = np.frombuffer(Imgpath.read(),np.uint8,offset=16).reshape(len(y_train),1,28,28)
with gzip.open(path_testLabel,'rb') as lbpath_test:
    y_test = np.frombuffer(lbpath_test.read(),np.uint8,offset=8)
with gzip.open(path_testImage, 'rb') as Imgpath_test:
    x_test = np.frombuffer(Imgpath_test.read(), np.uint8, offset=16).reshape(len(y_test),1,28,28)

import torch
# 将数据归一化处理
# 将图片信息转换数据类型
import numpy as np
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
# 归一化
x_train /= 255
x_test /= 255
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
x_train = torch.LongTensor(x_train)
x_test = torch.LongTensor(x_test)
# 将类别信息转换为one-hot编码形式(交叉熵损失不能用one-hot编码)
# y_train = torch.nn.functional.one_hot(y_train,10)
# y_test = torch.nn.functional.one_hot(y_test,10)



# /------------------ 数据处理--------------------*/


# /------------------ 模型建立--------------------*/
from torch.nn import Conv2d,MaxPool2d,Dropout,ReLU,Softmax,Linear,Sequential
# 类继承的方式（输入图片大小28*28*1）
class cnn(torch.nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.conv1 = Sequential(
            Conv2d(1,32,kernel_size=3,padding=1),#通道数1,输出通道32
            ReLU()
        )
        self.conv2 = Sequential(
            Conv2d(32,32,kernel_size=3,padding=0),
            ReLU()
        )
        self.conv3 = Sequential(
            Conv2d(32,64,kernel_size=3,padding=1),
            ReLU()
        )
        self.conv4 = Sequential(
            Conv2d(64,64,kernel_size=3,padding=0),
            ReLU()
        )
        self.maxpool = Sequential(
            MaxPool2d(kernel_size=2),
            Dropout(0.2)
        )
        self.dense1 = Sequential(
            Linear(1600,512),
            ReLU(),
            Dropout(0.5)
        )
        self.flatten = torch.nn.Flatten()
        self.dense2 = Sequential(
            Linear(512,10),
            Softmax()
        )
    #     输入x(28*28*1)
    def forward(self, x):
        x = x.float()
        # x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)#(28*28*32)维度变化
        x = self.conv2(x)#(26*26*32)
        x = self.maxpool(x)#(13*13*32)

        x = self.conv3(x)#(13*13*64)
        x = self.conv4(x)#(11*11*64)
        x = self.maxpool(x)#(5*5*64)

        x = self.flatten(x)#(1600)
        x = self.dense1(x)#(1600_-> 512)
        result = self.dense2(x)#(512->10)
        return result
model = cnn()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()


# /------------------ 模型建立--------------------*/


# /------------------ 模型训练--------------------*/
epoch = 5
import matplotlib.pyplot as plt
for i in range(epoch):
    # 调用模型预测（有点warning可以正常运行，torch版本太高容易报softmax的warning）
    x_train = torch.LongTensor(x_train)
    # Expected object of scalar type Long but got scalar type Float
    # 已经转换为了long型了
    y_pred = model(x_train)
    # 求损失
    loss = loss_fn(y_pred,y_train)
    # 梯度置零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 权值更新
    optimizer.step()
#     画出预测值和真实值之间的区别
#     print(i, loss.item())
#
#     plt.plot(i, loss.item())
#     plt.scatter(i, loss.item())
# plt.show()
