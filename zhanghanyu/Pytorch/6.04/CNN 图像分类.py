# ----------------开发者信息----------------------------
# 开发者：张涵毓
# 开发日期：2020年6月4日
# 内容：CNN-图像分类
# 修改内容：
# 修改者：
# ----------------开发者信息----------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 导入 pytorch, numpy, functools, os 和 gzip的包
# 2、读取数据与数据预处理
# 3、搭建CNN模型
# 4、训练模型，显示运行结果
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelBinarizer
from torch.nn import ReLU, Softmax, utils
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第3块显卡
#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、读取数据与数据预处理 -------------------------------

# 数据集和代码放一起即可
def load_data():
    paths = [
        'D:\\研究生\\代码\\Keras代码\\2.Convolutional Neural Networks(CNN 卷积神经网络)\\train-labels-idx1-ubyte.gz', 'D:\\研究生\\代码\\Keras代码\\2.Convolutional Neural Networks(CNN 卷积神经网络)\\train-images-idx3-ubyte.gz',
        'D:\\研究生\\代码\\Keras代码\\2.Convolutional Neural Networks(CNN 卷积神经网络)\\t10k-labels-idx1-ubyte.gz', 'D:\\研究生\\代码\\Keras代码\\2.Convolutional Neural Networks(CNN 卷积神经网络)\\t10k-images-idx3-ubyte.gz'
    ]
# 解压文件，划分成数据集
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

#加载数据
(x_train, y_train), (x_test, y_test) = load_data()
#设置参数
batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'

'''
# Convert class vectors to binary class matrices. 
   将类向量转换为二进制类矩阵
        类别独热编码
keras中to_categorical把类别标签转换为one-hot编码
to_categorical(y, num_classes=None, dtype='float32')
将整型标签转为onehot。y为int数组，num_classes为标签类别总数，大于max(y)（标签从0开始的）。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''
y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

#  -------------------------- 2、读取数据与数据预处理 -------------------------------

#  -------------------------- 3、搭建CNN模型 -------------------------------
class CNNic(nn.module):
    def __init__(self):
        super (CNNic,self).__init__()
        self.con1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.relu=nn.ReLU
        self.con2=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3)
        self.pooling=nn.MaxPool2d(kernel_size=2)
        self.drop1=nn.Dropout(0.25)
        self.con3=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.con4=nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3)
        self.flatten=nn.Flatten()
        self.dense1=nn.Linear(512)
        self.drop2=nn.Dropout(0.5)
        self.dense2=nn.Linear(512,10)
        self.soft=nn.Softmax()
    def forward(self,x):
        x=x.permute(0, 3, 1, 2)
        x=self.con1(x)
        x=self.relu(x)
        x=self.con2(x)
        x=self.relu(x)
        x=self.pooling(x)
        x=self.drop1(x)
        x=self.con3(x)
        x=self.relu(x)
        x=self.con4(x)
        x=self.relu(x)
        x=self.pooling(x)
        x=self.drop2(x)
        x=self.flatten(x)
        x=self.dense1(x)
        x=self.relu(x)
        x=self.dense2(x)
        x=self.soft(x)
        return x

model = CNNic()

# initiate RMSprop optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()


#  -------------------------- 3、搭建传统CNN模型 -------------------------------

#  -------------------------- 4、训练 -------------------------------
print("-----------训练开始-----------")
epoch = 5
for i in range(epoch):
    # 预测结果
    y_pred = model(x_train)
    # 计算损失
    loss = loss_func(y_pred, y_train)
    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 梯度更新
    optimizer.step()
    print(i,loss.item())
print("-----------训练结束-----------")
torch.save(model.state_dict(), "torch_image2.pkl")  # 保存模型参数
#  -------------------------------模型训练及保存------------------------