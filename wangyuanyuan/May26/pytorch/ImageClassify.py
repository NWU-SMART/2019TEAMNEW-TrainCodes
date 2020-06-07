#--------------------------------------开发者信息---------------------------------
#开发人：王园园
#开发日期：2020.5.26
#开发软件：pycharm
#开发项目:图像分类（pytorch）,相比keras只改了搭建框架部分
#注：有错误希望指正

#--------------------------------------代码布局-----------------------------------
#1、导入包
#2、数据导入
#3、数据预处理
#4、构建模型
#5、训练模型
#6、保存模型、显示运行结果

#---------------------------------------导入包---------------------------------------------
import gzip
import numpy as np
import os
from catalyst.utils import torch
from numpy import shape
from prompt_toolkit.input import Input

#----------------------------------------数据导入及数据处理----------------------------------
def loadData():
    paths = [
        'D:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'D:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:                #解压paths[1]压缩包，取出训练数据的标签
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:               #解压paths[1]压缩包，取出训练数据
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2], 'rb') as lbpath:                #解压paths[2]压缩包，取出测试数据的标签
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:                #解压paths[3]压缩包，取出测试数据
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train),(x_test, y_test) = loadData()
batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'

#对标签进行独热编码
y_train = torch.utils.to_categorical(y_train, num_classes)
y_test = torch.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255   #归一化
x_test /= 255    #归一化

#------------------------------------------搭建传统CNN模型-------------------------------
input = Input(shape())
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('conv1', torch.nn.Conv2d(1, 32, 3, 1, 1))
        self.conv.add_module('relu1', torch.nn.ReLU())
        self.conv.add_module('conv2', torch.nn.Conv2d(32, 32, 3, 1, 0))
        self.conv.add_module('relu2', torch.nn.ReLU())
        self.conv.add_module('maxpool1', torch.nn.MaxPool2d(2))
        self.conv.add_module('dropout1', torch.nn.Dropout(0.25))
        self.conv.add_module('conv3', torch.nn.Conv2d(32, 64, 3, 1, 0))
        self.conv.add_module('relu3', torch.nn.ReLU())
        self.conv.add_module('conv4', torch.nn.Conv2d(64, 64, 3, 1, 0))
        self.conv.add_module('relu4', torch.nn.ReLU())
        self.conv.add_module('maxpool2', torch.nn.MaxPool2d(2))la
        self.conv.add_module('dropout2', torch.nn.Dropout(0.25))
        self.conv.add_module('ftten', torch.nn.Flatten())
        self.conv.add_module('dense1', torch.nn.Linear(1600, 512))
        self.conv.add_module('relu5', torch.nn.ReLU())
        self.conv.add_module('dropout3', torch.nn.Dropout(0.5))
        self.conv.add_module('dense2', torch.nn.Linear(512, 10))
        self.conv.add_module('softmax', torch.nn.Softmax())

    def forward(self, input):
        x = self.conv(input)
        return x

model = model()



