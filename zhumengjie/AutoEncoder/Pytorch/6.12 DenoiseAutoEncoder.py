# 开发者：朱梦婕
# 开发日期：2020年6月12日
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
# 数据放到本地路径

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

# 加入噪声数据

noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

x_train = torch.Tensor(X_train)
x_test = torch.Tensor(X_test)

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

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
        x = x.permute(0, 3, 1, 2)  # input[60000, 28, 28, 1]---->[6000,1,28,28] 交换位置
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

