# ----------------开发者信息--------------------------------#
# 开发者：高强
# 开发日期：2020年5月27日
# 开发框架：pytorch
# 注意事项： 要在服务器上跑，不然卡死你的电脑
#-----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入相关包
# 2、加载图像数据
# 3、图像数据预处理
# 4、训练模型
# 5、保存模型与模型可视化
# 6、训练过程可视化
#--------------------------------------------------------------#

#---------------------------------------------------加载图像数据-----------------------------------------------------#
import numpy as np
import gzip            # 使用python gzip库进行文件压缩与解压缩
def load_data():
    # 训练标签 训练图像 测试标签 测试图像
    # 本地
    # paths = [
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\train-labels-idx1-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\train-images-idx3-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-labels-idx1-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    # ]
    # 服务器
    paths = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
    ]
    # 读取训练标签(解压)
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取训练图像(解压)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    # 读取测试标签(解压)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取测试图像(解压)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
# 调用函数 获取训练数据和测试数据
(x_train, y_train), (x_test, y_test) = load_data()


#-------------------------------------------图像数据预处理-----------------------------------------------------------#
# #  将整型的类别标签转为onehot编码
# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# y_train = encoder.fit_transform(y_train) # one-hot编码
# y_train = np.array(y_train)             # 放在数组里
# y_test = encoder.fit_transform(y_test)  # one-hot编码
# y_test = np.array(y_test)               # 放在数组里
# 注: 不能使用one-hot编码，否则在交叉熵函数那里会报错
import torch
from torch.autograd import Variable
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # 数据归一化
x_test /= 255   # 数据归一化
x_train = Variable(torch.from_numpy(x_train))  # x_train变为variable数据类型
x_test = Variable(torch.from_numpy(x_test))    # x_test变为variable数据类型
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------构建CNN模型----------------------------------------------------------#
##方法一：class ##
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Conv1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size =3,stride=1,padding=1) #1*28*28-32*28*28
        self.relu1 = torch.nn.ReLU()
        self.Conv2 = torch.nn.Conv2d(32,32,3,1,0)   # 32*28*28-32*26*26
        self.relu2 = torch.nn.ReLU()
        self.MaxPool1 = torch.nn.MaxPool2d(2)       # 32*26*26-32*13*13
        self.Dropout1 = torch.nn.Dropout(0.25)

        self.Conv3 = torch.nn.Conv2d(32,64,3,1,1)   # 32*13*13-64*13*13
        self.relu3 = torch.nn.ReLU()
        self.Conv4= torch.nn.Conv2d(64,64,3,1,1)    # 64*13*13-64*13*13
        self.relu4 = torch.nn.ReLU()
        self.MaxPool2 = torch.nn.MaxPool2d(2)       # 64*13*13-64*6*6
        self.Dropout2 = torch.nn.Dropout(0.25)

        self.Flatten = torch.nn.Flatten() # 拉平
        self.linear1 = torch.nn.Linear(2304,512)
        self.relu2 = torch.nn.ReLU()
        self.Dropout3 = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(512, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self,x):
        # 解决RuntimeError问题: Given groups=1, weight of size 32 1 3 3, expected input[60000, 28, 28, 1] to have 1 channels, but got 28 channels instead
        x = x.permute(0,3,1,2) #  input[60000, 28, 28, 1]---->[6000,1,28,28] 交换位置
        x = self.Conv1(x)
        x = self.relu1(x)
        x = self.Conv2(x)
        x = self.relu2(x)
        x = self.MaxPool1(x)
        x = self.Dropout1(x)

        x = self.Conv3(x)
        x = self.relu3(x)
        x = self.Conv4(x)
        x = self.relu4(x)
        x = self.MaxPool2(x)
        x = self.Dropout2(x)

        x = self.Flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.Dropout3(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

model = Model()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()
Epoch = 5
## 开始训练 ##
for t in range(Epoch):

    x = model(x_train)          # 向前传播
    loss = loss_fn(x, y_train)  # 计算损失

    if (t + 1) % 1 == 0:        # 每训练1个epoch，打印一次损失函数的值
        print(loss.item())
    if (t + 1) % 5 == 0:
        torch.save(model.state_dict(), "./model.pkl")  # 每5个epoch保存一次模型
        print("save model")

    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度








