# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年5月25日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、数据载入与预处理 -------------------------------
path = 'D:\\keras_datasets\\mnist.npz'# 数据地址
f = np.load(path)#载入数据
X_train=f['x_train']# 获取训练数据
X_test=f['x_test']# 获取测试数据
f.close()

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) #将训练数据reshape为28x28x1
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1) #将测试数据reshape为28x28x1

X_train = X_train.astype("float32")/255.#归一化
X_test = X_test.astype("float32")/255.

X_train = Variable(torch.from_numpy(X_train)).float()# 数据变为variable类型
X_test = Variable(torch.from_numpy(X_test)).float()

X_train =X_train.permute(0,3,2,1) # 改变数据channel位置，否则输入模型报错,变成1x28x28
X_test =X_test.permute(0,3,2,1)
#  -------------------------- 2、数据载入与预处理 -------------------------------

#  -------------------------- 3、模型训练   --------------------------------
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential( # 定义编码层
            nn.Conv2d(1, 16, 3, stride=3, padding=1),# 16x10x10
            nn.ReLU(),# 激活函数
            nn.MaxPool2d(2, stride=2),# 16x5x5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),# 8x3x3
            nn.ReLU(),# 激活函数
            nn.MaxPool2d(2, stride=1)# 8x2x2
        )
        self.decoder = nn.Sequential( # 定义解码层
            nn.ConvTranspose2d(8, 16, 3, stride=2), #16x5x5
            nn.ReLU(),# 激活函数
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),# 8x15x15
            nn.ReLU(),# 激活函数
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1), # 1x28x28
            nn.Sigmoid())# 激活函数
    def forward(self, x):
        encode = self.encoder(x)# 编码层
        decode = self.decoder(encode)# 解码层
        return decode

model = autoencoder()# 定义model
loss_fn = nn.BCELoss() #损失函数
learning_rate = 1e-4 # 学习率
EPOCH = 5  # epoch,迭代多少次
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #SGD优化器

for epoch in range(5):
    output = model(X_train)# 输入训练数据，获取输出
    loss = loss_fn(output, X_train)# 输出和训练数据计算损失函数
    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播
    optimizer.step()#梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 5, loss.item()))#每训练1个epoch，打印一次损失函数的值
#  -------------------------- 3、模型训练   --------------------------------
