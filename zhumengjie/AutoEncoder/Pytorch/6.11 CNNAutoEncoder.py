# 开发者：朱梦婕
# 开发日期：2020年6月11日
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

# 数据预处理
# 数据格式进行转换
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

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

x_train = torch.Tensor(X_train)
x_test = torch.Tensor(X_test)

#  --------------------- 读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、构建CNN自编码器模型 ---------------------
class CNNAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__()
        # 编码器
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 解码器
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.Upsample((8, 8)),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),

            nn.Upsample((16, 16)),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0, stride=1),
            nn.ReLU(),

            nn.Upsample((28, 28)),

            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = CNNAutoEncoder()
#  -------------------------- 4、模型训练 -------------------------------

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_func = torch.nn.BCELoss()
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

