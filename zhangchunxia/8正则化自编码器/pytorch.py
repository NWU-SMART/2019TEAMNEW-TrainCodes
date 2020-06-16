# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月16日
# 修改日期：
# 修改人：
# 修改内容：
#备注：可以运行出来
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 pytorch的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、训练模型
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import numpy as np
import torch
import torch.nn as nn
import os
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
path = 'D:\\northwest\\小组视频\\5单层自编码器\\mnist.npz'
f = np.load(path)
X_train = f['x_train']
X_test = f['x_test']
f.close()
print(X_train.shape)
print(X_test.shape)
X_train = X_train.astype('float32')/255#数据预处理，归一化将像素点转换为[0,1}之间
X_test = X_test.astype('float32')/255
# np.prod是将28X28矩阵转化成1X784向量，方便BP神经网络输入层784个神经元读取
X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))#60000*784
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))#10000*784
X_train = torch.Tensor(X_train)  # 转换为tenser
X_test = torch.Tensor(X_test)    # 转换为tenser
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建自编码器模型 ------------------------------
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = torch.nn.Sequential(#编码
            torch.nn.Linear(784,32),
            torch.nn.ReLU(),)
        self.decoder = torch.nn.Sequential(#解码
            torch.nn.Linear(32,784),
            torch.nn.Sigmoid()
        )
    def forward(self,x):
       encode = self.encoder(x)
       decode = self.decoder(encode)
       return decode
model = autoencoder()
#  ---------------------  3、构建自编码器模型 ------------------------------
#  ---------------------- 4、模型训练 ----------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #使用SGD优化器
loss_fn = torch.nn.MSELoss()
for i in range(15):
    X_P =model(X_train)#前向传播
    MSEloss = loss_fn(X_P,X_train)
    regularization_loss = 0
    for parm in model.parameters():#定义l1正则项
        regularization_loss += torch.sum(torch.abs(parm))
        loss = MSEloss+0.01*regularization_loss    #损失函数加上惩罚项成为真正的损失函数
    if(i+1)%1==0:
            print('epoch [{}/{}],loss:{:.4f}'.format(i + 1, 15, loss.item()))  # 每迭代一次，打印一次损失函数的值
    optimizer.zero_grad()#梯度清零,在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()#反向传播
    optimizer.step()#梯度更新

#  --------
