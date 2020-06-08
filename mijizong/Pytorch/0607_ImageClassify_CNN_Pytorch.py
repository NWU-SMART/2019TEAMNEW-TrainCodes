# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/7
#  @Author: MiJizong
#  @Content: 图像分类CNN——Pytorch方法实现
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: 参照亚楠的程序，修改程序运行部分代码，使数据能在GPU上小批量处理
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： --------------------------------------
# 1、导入 torch, matplotlib, numpy, gzip 和 os的包
# 2、读取数据与数据预处理
# 3、构建Sequential模型
# 4、模型训练与输出
# ----------------------   代码布局： --------------------------------------

#  -------------------------- 1、导入需要包 --------------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import gzip
import numpy as np
import torch.utils.data as Data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块显卡
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
#  -------------------------- 1、导入需要包 ---------------------------------


#  -------------------------- 2、读取数据与数据预处理 ------------------------

# 数据集和代码放一起即可
def load_data():
    paths = [
        'D:\\Office_software\\PyCharm\\datasets\\train-labels-idx1-ubyte.gz',
        'D:\\Office_software\\PyCharm\\datasets\\train-images-idx3-ubyte.gz',
        'D:\\Office_software\\PyCharm\\datasets\\t10k-labels-idx1-ubyte.gz',
        'D:\\Office_software\\PyCharm\\datasets\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:                       # 解压paths[0]中的数据，并取出训练标签
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)  # 参数offset为读取的起始位置，默认为0

    with gzip.open(paths[1], 'rb') as imgpath:                      # 解压paths[1]中的数据，并取出训练数据
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)  # 28 * 28

    with gzip.open(paths[2], 'rb') as lbpath:                       # 解压paths[2]中的数据，并取出测试标签
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:                      # 解压paths[3]中的数据，并取出测试数据
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.astype('float32')  # 转换数据类型
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

# 转换为variable数据类型    交叉熵损失不能用one-hot编码
x_train = Variable(torch.from_numpy(x_train))
x_test = Variable(torch.from_numpy(x_test))
# 转换为tensor形式
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
#  -------------------------- 2、读取数据与数据预处理 -------------------------

#  -------------------------- 3、构建Sequential模型 --------------------------
class ImageClassify(nn.Module):
    def __init__(self):
        super(ImageClassify,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),  # 输入通道1，输出通道32，3x3的卷积核，步长1，padding 1
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(1600,512),  # 全连接层
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,10),
            nn.Softmax())

    def forward(self,x):
        x = x.permute(0,3,1,2)  #  input[60000, 28, 28, 1]---->[6000,1,28,28] 交换维度位置
        # ↑ 解决 RuntimeError: Given groups=1, weight of size [32, 1, 3, 3], expected input[60000, 28, 28, 1] to have 1 channels, but got 28 channels instead
        x = self.layer(x)
        return x

model = ImageClassify()
print(model)
'''# 以下三行可以调用GPU加速训练，也就是在模型，x_train，y_train后面加上cuda()'''
model = model.cuda()
x_train = x_train.cuda()
y_train = y_train.cuda()
#  -------------------------- 3、构建Sequential模型 --------------------------

#  -------------------------- 4、模型训练与输出 -------------------------------
loss_func = nn.CrossEntropyLoss()                        # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)  # Adam优化器

#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)

loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (x_train, y_train) in enumerate(loader):
        train_prediction = model(x_train)
        loss = loss_func(train_prediction, y_train)  # 计算损失
        loss_list.append(loss)       # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()        # 梯度清零
        loss.backward()              # 反向传播
        optimizer.step()             # 参数更新
        running_loss += loss.item()  # 损失叠加
    else:
        print(f"第{t}代训练损失为：{running_loss/len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
#  -------------------------- 4、模型训练与输出 -------------------------------
