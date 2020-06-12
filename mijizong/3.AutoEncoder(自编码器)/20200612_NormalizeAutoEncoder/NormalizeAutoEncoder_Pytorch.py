# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/12
#  @Author: MiJizong
#  @Content: 正则自编码器——Pytorch方法实现
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： ----------------------
# 1、导入相应的包
# 2、读取手写体数据及与图像预处理
# 3、构建正则自编码器模型
# 4、训练
# 5、可视化
# 6、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from torch.utils.data import TensorDataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块显卡
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
# 以上两句命令如果不添加汇报下列错误：
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
# That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do
# is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static
# linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you
# can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute,
# but that may cause crashes or silently produce incorrect results. For more information, please see
# http://www.intel.com/software/products/support/.
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)
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

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# 进行格式转变
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)

set = TensorDataset(X_train,X_train)  # 将数据集包装为TensorDataset
loader = DataLoader(dataset=set,batch_size=128,shuffle=False)  # 使用dataloader类


input_size = 784
hidden_size = 32
output_size = 784
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、构建正则自编码器Sequential模型 ----------------

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,output_size),
            nn.Sigmoid())

    def forward(self,x):
        x = self.layer(x)
        return x

autoencoder = Autoencoder()
print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(),lr = 1e-4)
loss = nn.MSELoss()

#  --------------------- 3、构建正则自编码器Sequential模型 ----------------


#  --------------------- 4、训练 -----------------------------------------

loss_total = []
epoch_total = []

epochs = 5
for epoch in range(epochs):
    noregular_loss = 0.0
    train_loss = 0.0                                                # 初始化损失函数
    for i,data in enumerate(loader):                                # 挨个batch训练
        regularization_loss = 0
        for param in autoencoder.parameters():                      # L1正则化
            regularization_loss += torch.sum(torch.abs(param))
        pre = autoencoder(data[0])                                  # data[0]为训练图像
        batch_loss = loss(pre,data[1])                              # data[1]为真实图像
        loss_regularize = batch_loss + 10e-5 * regularization_loss
        optimizer.zero_grad()                                       # 梯度清零
        loss_regularize.backward()                                  # 损失函数反向传播
        optimizer.step()                                            # 优化器更新
        noregular_loss += batch_loss
        train_loss += loss_regularize                               # 一个epoch内的所有loss相加
    print('epoch %3d, noregular % 10f, train_loss %10f'%(
        epoch+1,noregular_loss / len(loader),train_loss / len(loader)))
    loss_total.append(train_loss/len(loader))                       # 每个epoch的loss相加
    epoch_total.append(epoch)

#  --------------------- 4、训练 -----------------------------------------


#  --------------------- 5、可视化 ---------------------------------------

# decoded_imgs 为输出层的结果
pre = autoencoder(X_test).detach().numpy()

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i+1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # 解码效果图
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(pre[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

#  --------------------- 5、可视化 ---------------------------------------


#  --------------------- 6、训练过程可视化 --------------------------------

# 画出Train_loss曲线
plt.plot(epoch_total,loss_total,label='loss')
plt.title('autoencoder loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()

#  --------------------- 6、训练过程可视化 ---------------------------------