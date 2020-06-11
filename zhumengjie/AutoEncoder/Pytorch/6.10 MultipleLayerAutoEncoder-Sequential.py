# 开发者：朱梦婕
# 开发日期：2020年6月10日
# 开发框架：pytorch
#----------------------------------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入 torch, matplotlib, numpy
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型训练
# 5、模型测试
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
#  --------------------------导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'E:\\study\\kedata\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 数据放到本地路径test

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
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

x_train = torch.FloatTensor(X_train)
x_test = torch.FloatTensor(X_test)

#  --------------------- 读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、构建单层自编码器模型 ---------------------

# 输入、隐藏和输出层神经元个数 (3个隐藏层)
input_size = 784
hidden_size = 128
code_size = 64

# 搭建模型
class MAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(MAutoEncoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=code_size),
            nn.ReLU(),
            nn.Linear(in_features=code_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=input_size),
            nn.Softmax()
           )

    def forward(self, x):
        x = self.layer(x)
        return x

model = MAutoEncoder()
print(model)  # 打印网络层次结构

#  -------------------------- 4、模型训练 -------------------------------

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_func = torch.nn.MSELoss()
print("-----------训练开始-----------")

epoch = 5
for i in range(epoch):
    # 预测结果
    pred = model(x_train)
    # 计算损失
    loss = loss_func(pred, x_train)
    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 梯度更新
    optimizer.step()
    print(i, loss.item())

print("-----------训练结束-----------")
torch.save(model.state_dict(), "torch_MAutoEncode.pkl")  # 保存模型参数
# -------------------------------模型训练------------------------

#  -------------------------- 5、模型测试 -------------------------------
print("-----------测试开始-----------")

model.load_state_dict(torch.load('torch_MAutoEncode.pkl')) # 加载训练好的模型参数
epoch = 5
for i in range(epoch):
    # 预测结果
    pred = model(x_test)
    # 计算损失
    loss = loss_func(pred, x_test)
    # 打印迭代次数和损失
    print(i, loss.item())

    # 打印图片显示decoder效果
    pred = pred.detach().numpy()
    plt.imshow(pred[i].reshape(28, 28))
    plt.gray()  # 显示灰度图像
    plt.show()

print("-----------测试结束-----------")
# -------------------------------模型测试------------------------

