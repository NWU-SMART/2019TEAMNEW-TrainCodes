# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/10
#  @Author: MiJizong
#  @Content: 多层自编码器——Pytorch三种方法实现
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相应的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器三种模型
# 4、训练
# 5、模型可视化
# 6、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 8、训练过程可视化
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os
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
X_train = f['x_train']
# 测试数据
X_test = f['x_test']
f.close()
# 数据放到本地路径

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)  # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

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

# 输入、隐藏和输出层神经元个数 (3个隐藏层)
input_size = 784
hidden_size = 128
code_size = 64  # dimension 784 = (28*28) --> 128 --> 64 --> 128 --> 784 = (28*28)

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3.1、构建多层自编码器Sequential模型 ----------------
class Autoencoder1(nn.Module):
    def __init__(self):
        super(Autoencoder1,self).__init__()
        self.layer = nn.Sequential(
                        nn.Linear(input_size,hidden_size),  # 748 → 128
                        nn.ReLU(),
                        nn.Linear(hidden_size,code_size),  # 128 → 64
                        nn.ReLU(),
                        nn.Linear(code_size, hidden_size),  # 64 → 128
                        nn.ReLU(),
                        nn.Linear(hidden_size, input_size),  # 128 → 748
                        nn.Sigmoid())

    def forward(self,x):
        x = self.layer(x)
        return x

autoencoder1 = Autoencoder1()
print(autoencoder1)

#  --------------------- 3.1、构建多层自编码器Sequential模型 ----------------

#  --------------------- 3.2、构建多层自编码器add_module方法 ----------------
class Autoencoder2(torch.nn.Module):
    def __init__(self):
        super(Autoencoder2, self).__init__()
        self.layer = torch.nn.Sequential()
        self.layer.add_module('layer1', torch.nn.Linear(input_size, hidden_size))  # 748 → 128
        self.layer.add_module('relu', torch.nn.ReLU())
        self.layer.add_module('layer2', torch.nn.Linear(hidden_size, code_size))  # 128 → 64
        self.layer.add_module('relu', torch.nn.ReLU())
        self.layer.add_module('layer1', torch.nn.Linear(code_size, hidden_size))  # 64 → 128
        self.layer.add_module('relu', torch.nn.ReLU())
        self.layer.add_module('layer2', torch.nn.Linear(hidden_size, input_size))  # 128 → 748
        self.layer.add_module('sigmoid', torch.nn.Sigmoid())

    def forward(self, x):  # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出 #
        x = self.layer(x)
        return x

autoencoder2 = Autoencoder2()  # 实例化
#print(autoencoder2)

#  --------------------- 3.2、构建多层自编码器add_module方法 ---------------

#  --------------------- 3.3、构建多层自编码器class继承模型 ----------------

class Autoencoder3(torch.nn.Module):
    def __init__(self):
        super(Autoencoder3, self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.rule1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size,code_size)
        self.rule2 = nn.ReLU()
        self.layer3 = nn.Linear(code_size, hidden_size)
        self.rule3 = nn.ReLU()
        self.layer4 = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.layer1(x)
        x = self.rule1(x)
        x = self.layer2(x)
        x = self.rule2(x)
        x = self.layer3(x)
        x = self.rule3(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        return x

autoencoder3 = Autoencoder3()  # 实例化
#print(autoencoder3)

#  --------------------- 3.3、构建多层自编码器class继承模型 ----------------

#  ----------------------- 4、模型训练与输出 ------------------------------
'''# 以下三行可以调用GPU加速训练，也就是在模型，x_train，y_train后面加上cuda()'''
autoencoder1 = autoencoder1.cuda()
X_train = X_train.cuda()
X_test = X_test.cuda()

loss_func = torch.nn.MSELoss()                                   # 损失函数
optimizer = torch.optim.Adam(autoencoder1.parameters(),lr=1e-4)  # Adam优化器

#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
torch_dataset = Data.TensorDataset(X_train, X_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)

loss_list = []  # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (X_train, X_train) in enumerate(loader):
        train_prediction = autoencoder1(X_train)     # 一轮训练
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


