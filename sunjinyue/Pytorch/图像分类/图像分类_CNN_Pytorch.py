# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月8日
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import functools
import gzip
import matplotlib.pyplot as plt
import keras
import numpy as np
import os
import torch
import torch.nn as nn

#  -------------------------- 读取的载入和预处理 -------------------------------

def load_data():
    paths = [
        'D:\\应用软件\\研究生学习\\train-labels-idx1-ubyte.gz',
        'D:\\应用软件\\研究生学习\\train-images-idx3-ubyte.gz',
        'D:\\应用软件\\研究生学习\\t10k-labels-idx1-ubyte.gz',
        'D:\\应用软件\\研究生学习\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:           #  'rb'  以二进制格式打开一个文件用于只读。
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)   #frombuffer将data以流的形式读入转化成ndarray对象

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 128
# num_classes = 10
epochs = 3
# data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')       # os.getcwd() 返回当前目录         os.path.join()地址拼接
model_name = 'keras_fashion_trained_model.h5'

# Convert class vectors to binary class matrices. 类别独热编码
# y_train = keras.utils.to_categorical(y_train, num_classes)        #(编码矩阵，编码长度)
# y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(x_train.shape[1:])  # (28, 28, 1)
print(x_train.shape)  # (60000, 28, 28, 1)
print(y_train.shape)

# --------------------模型搭建CNN---------------------------

class CNN_Image_Model(nn.Module):
    def __init__(self):
        super(CNN_Image_Model,self).__init__()
        self.conv2d1 = nn.Conv2d(1, 3, (3, 3), padding=1)  # 3*28*28
        self.relu1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(3, 6, (3, 3))  # 6*26*26
        self.relu2 = nn.ReLU()
        self.maxpooling2d1 = nn.MaxPool2d((2, 2))  # 6*13*13
        self.dropout1 = nn.Dropout(0.25)

        self.conv2d3 = nn.Conv2d(6, 16, (3, 3), padding=1)  # 16*13*13
        self.relu3 = nn.ReLU()
        self.conv2d4 = nn.Conv2d(16, 16, (3, 3), padding=1)  # 16*13*13
        self.relu4 = nn.ReLU()
        self.maxpooling2d2 = nn.MaxPool2d((2, 2))  # 16*6*6
        self.dropout2 = nn.Dropout()

        self.flatten = nn.Flatten()  # 2304
        self.fc1 = nn.Linear(576, 512)  # 512
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # 10
        self.softmax = nn.Softmax()

    def forward(self, x):

            x = x.permute(0, 3, 1, 2)                # [60000, 28, 28, 1]  ---> [60000, 1, 28, 28]  解决通道报错
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

model = CNN_Image_Model() #实例化模型

# ----------------------------------------------------------

# --------------------------模型训练-------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()
#  ---------------------------训练------------------------------
loss_list = [] #需要将loss存入列表中，所以建立一个列表
for t in  range(3):
    train_Prediction = model(x_train)
    loss = loss(train_Prediction,y_train) #带入损失函数计算损失
    loss_list.append(loss) #append加入list

    optimizer.zero_grad() #Pytorch 需要在每一个batch手动清零梯度
    loss.backward() #反向传播，算参数
    optimizer.step() #参数更新
    print(loss)

plt.plot(loss_list,'r')
plt.show()
