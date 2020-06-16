# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月16日
# 修改日期：
# 修改人：
# 修改内容：
#备注：好像要在服务器里面跑，我的电脑没跑出来
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 torch的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、训练模型
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
from torch.nn.modules import loss
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
path = 'D:\\northwest\\小组视频\\5单层自编码器\\mnist.npz'# 数据地址
f = np.load(path)#载入数据
X_train=f['x_train']# 获取训练数据
X_test=f['x_test']# 获取测试数据
f.close()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #将训练数据reshape为28x28x1
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) #将测试数据reshape为28x28x1
X_train = X_train.astype("float32")/255.#归一化,将像素点转换为[0,1]之间
X_test = X_test.astype("float32")/255.
# 加入噪声数据，生成噪声，在原始数据的基础上加0.5*均值为0，方差为1.0的正态分布
noise_factor = 0.5
X_train_noise = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape) # 正态分布
X_test_noise = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noise = np.clip(X_train_noise, 0., 1.) # X_train_noisy是数组，0.是将比0小的数替换为0，将数组中的元素限制在0-1之间
X_test_noisy = np.clip(X_test_noise, 0., 1.)#1.是将比1大的数替换为1
X_train = Variable(torch.from_numpy(X_train)).float()# 原始数据变为variable类型
X_test = Variable(torch.from_numpy(X_test)).float()
X_train_noise = Variable(torch.from_numpy(X_train_noise)).float()#将噪声数据转换为Variable类型
X_test_noise = Variable(torch.from_numpy(X_test_noise)).float()
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建自编码器模型 ------------------------------
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = nn.Sequential(
          nn.Conv2d(1, 32, 3, stride=3, padding=1),#参数strides，padding分别决定了卷积操作中滑动步长和图像边沿填充的方式。1*28*28--32*28*28
          nn.ReLU(),
          nn.MaxPool2d(2,stride=2),#32*28*28--32*14*14
          nn.Conv2d(32,32,3,stride=3,padding=1),#32*14*14---32*14*14
          nn.ReLU(),
          nn.MaxPool2d(2,stride=2),#32*14*14---32*7*7
         )
        self.decoder = nn.Sequential(
          nn.Conv2d(32,32,3,stride=3,padding=1),#32*7*7----32*7*7
          nn.ReLU(),
          nn.Upsample((14,14)),#32*7*7---32*14*14
          nn.Conv2d(32, 32, 3, stride=3, padding=1),#32*14*14---32*14*14
          nn.ReLU(),
          nn.Upsample((28,28)),#32*14*14---32*28*28
          nn.Conv2d(32, 1, 3, stride=3, padding=1),#32*28*28---1*28*28
          nn.Sigmoid()
      )
    def forward(self,x):
        encode=self.encoder(x)
        decode=self.decoder(encode)
        return decode
model = autoencoder()
#  ---------------------  3、构建自编码器模型 ------------------------------
#  ---------------------- 4、模型训练 ----------------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for i in range(5):
    X_P = model(X_train_noise)
    MSEloss = loss_fn(X_P,X_train_noise)
    if (i + 1) % 1 == 0:
        print('epoch [{}/{}],loss:{:.4f}'.format(i + 1, 15, loss.item()))  # 每迭代一次，打印一次损失函数的值
    optimizer.zero_grad()#梯度清零
    loss.backward()
    optimizer.step()#梯度更新
    #  ---------------------- 4、模型训练 ----------------------------------------
