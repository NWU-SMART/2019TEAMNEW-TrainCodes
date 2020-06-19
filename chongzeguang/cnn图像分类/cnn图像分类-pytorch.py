# ----------------开发者信息--------------------------------#
# 开发者：崇泽光
# 开发日期：2020年6月18日
# 修改日期：
# 修改人：
# 修改内容：
# coding: utf-8
# --------------------------------------------------------#

# 导入需要包
import torch.nn as nn
import torch
from torch.autograd import Variable
import torch
import gzip
import numpy as np # NumPy 是一个运行速度非常快的数学库，主要用于数组计算
import os # 操作系统接口模块


# 读取数据与数据预处理
def load_data():
    paths = [
        'D:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'D:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8) # numpy.frombuffer 用于实现动态数组，返回数组的数据类型，offset是读取的起始位置，默认为0。

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1) # 图片尺寸28*28

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # 数据归一化
x_test /= 255   # 数据归一化
x_train = Variable(torch.from_numpy(x_train)) # x_train变为variable数据类型
x_test = Variable(torch.from_numpy(x_test)) # variable数据类型
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# 确定超参数
batch_size = 32 # 一次训练所选取的样本数
num_classes = 10 # 一共有10种类型
epochs = 5 # 定义为向前和向后传播中所有批次的单次训练迭代。简单说，epochs指的就是训练过程中数据将被“轮”多少次
data_augmentation = True  # 图像增强，数据增强主要用来防止过拟合，用于dataset较小的时候。
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn') # 路径拼接os.path.join()函数；使用os.getcwd()函数获得当前的路径。
model_name = 'keras_fashion_trained_model.h5'


# 搭建CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 32, (3, 3), padding=1) # 2D卷积，很适合处理图像输入
        self.relu1 = nn.ReLU() # relu激活
        self.conv2d2 = nn.Conv2d(32, 32, (3, 3)) # 2d卷积
        self.relu2 = nn.ReLU() # relu激活
        self.maxpooling2d1 = nn.MaxPool2d((2, 2)) # 最大池化
        self.dropout1 = nn.Dropout(0.25) # dropout减少过拟合

        self.conv2d3 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.relu3 = nn.ReLU()
        self.conv2d4 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.relu4 = nn.ReLU()
        self.maxpooling2d2 = nn.MaxPool2d((2, 2))
        self.dropout2 = nn.Dropout()

        self.flatten = nn.Flatten() # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
        self.fc1 = nn.Linear(2304, 512) # 为model添加Dense层，即全链接层，512为输出
        self.relu5 = nn.ReLU() # relu激活
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        self.softmax = nn.Softmax() # 通过softmax激励函数进行分类

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv2d1(x)
        x = self.relu1(x)
        x = self.conv2d2(x)
        x = self.relu2(x)
        x = self.maxpooling2d1(x)
        x = self.dropout1(x)

        x = self.conv2d3(x)
        x = self.relu3(x)
        x = self.conv2d4(x)
        x = self.relu4(x)
        x = self.maxpooling2d2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
model = CNN()

# 训练参数设置
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # 优化器，lr (float, 可选)是学习率（默认：1e-3）
loss = nn.CrossEntropyLoss() # 损失函数，pytorch中的交叉熵损失

for t in range(epochs):
    pre = model(x_train) # 向前传播
    batch_loss = loss(pre, y_train) # 计算损失
    optimizer.zero_grad()# 梯度清零
    batch_loss.backward() # 反向传播
    optimizer.step() # 更新梯度
    if (t + 1) % 1 == 0:  # 每训练1个epoch，打印一次损失函数的值
        print(loss.item())

