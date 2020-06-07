# ----------------开发者信息-------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月4日
# 开发框架：pytorch
#---------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 pytorch, numpy, functools, os 和 gzip的包
# 2、参数定义
# 3、读取数据与数据预处理
# 4、搭建CNN模型
# 5、模型训练
#----------------------------代码布局------------------------#

#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelBinarizer
from torch.nn import ReLU, Softmax, utils
import os
# os.environ['CUDA_VISIBLE_DEVICES']='3,4'
# 使用四块显卡进行加速
#  --------------------------导入需要包 -------------------------------

#  -------------------------- 2、参数定义 -------------------------------
batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'torch_image_trained_model2.h5'
#  -------------------------- 参数定义 -------------------------------

#  -------------------------- 3、读取数据与数据预处理 -------------------------------
# 函数：数据加载
def load_data():
    # 写入文件路径
    paths = [
        'E:\\study\\mnist\\train-labels-idx1-ubyte.gz', 'E:\\study\\mnist\\train-images-idx3-ubyte.gz',
        'E:\\study\\mnist\\t10k-labels-idx1-ubyte.gz', 'E:\\study\\mnist\\t10k-images-idx3-ubyte.gz'
    ]

    # 将文件解压并划分为数据集
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data() # 加载数据集

# 将类型信息进行one-hot编码(10类)
y_train = LabelBinarizer().fit_transform(y_train)# 对y进行one-hot编码
y_test = LabelBinarizer().fit_transform(y_test)

# 将图片信息转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化
x_train /= 255
x_test /= 255

# 转换为tensor形式
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
#  -------------------------- 读取数据与数据预处理 -------------------------------

#  -------------------------- 4、搭建CNN模型 -------------------------------
class CNNmodel_image2(nn.Module):
    def __init__(self):
        super(CNNmodel_image2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.25)
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.layer3(x)
        return output

model = CNNmodel_image2()
print(model)  # 打印网络层次结构
#  -------------------------- 搭建CNN模型 -------------------------------

#  -------------------------- 5、模型训练及保存 -------------------------------
# model = model.cuda()
# x_train = x_train.cuda()
# y_train = y_train.cuda()
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()
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