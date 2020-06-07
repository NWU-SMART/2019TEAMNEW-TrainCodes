# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年5月26日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、数据载入与预处理 -------------------------------
path = 'D:\\keras_datasets\\mnist.npz'# 数据地址
f = np.load(path) #载入数据
X_train=f['x_train'] # 获取训练数据
X_test=f['x_test'] # 获取测试数据
f.close()

X_train = X_train.astype("float32")/255.#归一化
X_test = X_test.astype("float32")/255.

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))#将28x28变为784，方便输入模型
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

X_train = Variable(torch.from_numpy(X_train)).float()#将数据转为variable类型
X_test = Variable(torch.from_numpy(X_test)).float()
#  -------------------------- 2、数据载入与预处理 -------------------------------

#  -------------------------- 3、模型训练以及保存   --------------------------------
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(# 定义编码层
            nn.Linear(784, 32),# 784---->32
            nn.ReLU())# 激活函数
        self.decoder = nn.Sequential(# 定义解码层
            nn.Linear(32,784),# 32----->784
            nn.Sigmoid())#激活函数
    def forward(self, x):
        encode = self.encoder(x) # 编码层
        decode = self.decoder(encode)# 解码层
        return decode
model = autoencoder()# 定义model
loss_fn = nn.MSELoss() #损失函数
learning_rate = 1e-4 # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #SGD优化器

for epoch in range(10):
    output = model(X_train)# 输入训练数据，获取输出

    regularization_loss =0 
    for param in model.parameters():# 定义L1正则项
        regularization_loss += torch.sum(torch.abs(param))

    a_loss = loss_fn(output, X_train)# 输出和训练数据计算损失函数
    loss = a_loss+0.01*regularization_loss  # 损失函数加上惩罚项成为真正的损失函数

    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播
    optimizer.step()#梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 10, loss.item()))#每训练1个epoch，打印一次损失函数的值
#  -------------------------- 3、模型训练以及保存   --------------------------------


