# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/11
#  @Author: MiJizong
#  @Content: 卷积自编码器——Pytorch实现
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相关的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、4、模型训练与输出
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
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
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 数据放到本地路径

# 数据格式进行转换
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#  数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
# 输出X_train和X_test维度
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28, 1)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 转换为tensor格式
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)

X_train = X_train.permute(0, 3, 1, 2)  # 维度顺序转换为1*28*28
X_test =X_test.permute(0, 3, 1, 2)
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、构建卷积自编码器模型 ----------------------------

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16, kernel_size=3,padding=1),  # 1*28*28 -> 16*28*28  输入通道1，输出通道16，3x3的卷积核，步长1，padding 1
            nn.ReLU(),
            nn.MaxPool2d((2,2),padding=1),                                      # 16*28*28 -> 16*14*14
            nn.Conv2d(in_channels=16,out_channels=8, kernel_size=3,padding=1),  # 16*14*14 -> 8*14*14
            nn.ReLU(),
            nn.MaxPool2d((2,2),padding=1),                                      # 8*14*14 -> 8*7*7
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1), # 8*7*7
            nn.ReLU(),
            nn.MaxPool2d((2, 2), padding=1))                                    # 8*4*4

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3, padding=1),    # 8*4*4 -> 8*4*4
            nn.ReLU(),
            nn.Upsample((8,8)),                                                 # 8*4*4 -> 8*8*8   制定输出的大小
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1), # 8*8*8 -> 8*8*8
            nn.ReLU(),
            nn.Upsample((16, 16)),                                                # 8*8*8 -> 8*16*16
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),           # 8*16*16 -> 16*14*14
            nn.ReLU(),
            nn.Upsample((28, 28)),                                                # 16*14*14 -> 16*28*28
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),# 16*28*28 -> 1*28*28
            nn.Sigmoid())

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

autoencoder = Autoencoder()
print(autoencoder)

#  --------------------- 3、构建卷积自编码器模型 ------------------------------


#  -------------------------- 4、模型训练与输出 -------------------------------

'''# 以下三行可以调用GPU加速训练，也就是在模型，x_train，y_train后面加上cuda()'''
model = autoencoder.cuda()
X_train = X_train.cuda()

loss_func = nn.MSELoss()                        # 损失函数
optimizer = torch.optim.Adam(autoencoder.parameters(),lr=1e-4)  # Adam优化器

#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
torch_dataset = Data.TensorDataset(X_train, X_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)

loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (X_train, X_train) in enumerate(loader):
        train_prediction = autoencoder(X_train)
        loss = loss_func(train_prediction, X_train)  # 计算损失
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