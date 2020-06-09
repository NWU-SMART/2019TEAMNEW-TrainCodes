# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月9日
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 torch,numpy,matplotlib的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型训练
# 5、训练过程可视化
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 ---------------------
path = '...'
f=np.load(path)
X_train = f['x_train']
X_test = f['x_test']
f.close()
print(X_train.shape)
print(X_test.shape)
X_train =  X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
print('X_train.shape',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
X_train = X_train.reshape((len(X_train),np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test),np.prod(X_test.shape[1:])))
X_train = torch.from_numpy(X_train)#转换为tensor
X_test=torch.from_numpy(X_test)
#  ---------------------  2、读取手写体数据及与图像预处理 ---------------------
#  ---------------------  3、构建单层自编码器模型 ----------------------------
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.net=torch.nn.Sequential(
            torch.nn.Linear(784,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,784),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x=self.net(x)
        return x
#  ---------------------  3、构建单层自编码器模型 ----------------------------
#  -----------------------4、模型训练与过程可视化 ----------------------------
model = autoencoder()
print(model)
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = torch.nn.MSELoss()
epochs=5
for i in range(epochs):
    X = model(X_train)#前向传播
    loss = loss_fn(X,X_train)
    if(i+1)%1==0:
       print(loss.item())

    if(i+1)%5==0:
       torch.save(model.state_dict(), "./pytorch_SingleLayerAutoEncoder_model.pkl")  # 每5个epoch保存一次模型
       print("save model")

    optimizer.zero_grad()#梯度更新之前先使用optimier对象提供的清除已经积累的梯度
    loss.backward()#反向传播
    optimizer.step()#更新梯度
#  -----------------------4、模型训练与过程可视化 ----------------------------
