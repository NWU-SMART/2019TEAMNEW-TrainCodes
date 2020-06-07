# -------------------------------------------------开发者信息----------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.6.2
# 开发框架：pytorch
# 温馨提示：
#----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------代码布局-----------------------------------------------------------#
# 1、读取手写体数据及与图像预处理
# 2、构建自编码器模型
# 3、训练过程可视化
# 4、保存模型

#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------读取手写体数据及与图像预处理------------------------------------------------#
import numpy as np
# 载入数据
path = 'F:\\Keras代码学习\\keras\\keras_datasets\\mnist.npz'
f = np.load(path)
print(f.files) # 查看文件内容 ['x_test', 'x_train', 'y_train', 'y_test']
# 定义训练数据 60000个
x_train = f['x_train']
# 定义测试数据 10000个
x_test = f['x_test']
f.close()
# 打印训练数据的维度
print(x_train.shape)  # (60000, 28, 28)
# 打印测试数据的维度
print(x_test.shape)   # (10000, 28, 28)

# 数据预处理
# 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.
# np.load是将28*28的矩阵转换为1*784，方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))  # 60000*784
x_test  = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))     # 10000*784

import torch
x_train = torch.Tensor(x_train)  # 转换为tenser
x_test = torch.Tensor(x_test)    # 转换为tenser
#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------构建自编码器模型---------------------------------------------------------#

import torch.nn as nn
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 784),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output

model = myModel()



#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------训练过程可视化、保存模型-------------------------------------------------------#

print(model)
# torch.optim的优化器固定实现L2正则化，不能实现L1正则化。
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=10.0)# weight_decay=10.0，即正则化的权重lambda =10.0

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()
## 开始训练 ##
for t in range(15):

    x = model(x_train)          # 向前传播
    MSEloss = loss_fn(x, x_train)  # 计算损失
    # L1 Loss
    regularization_loss = 0
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    loss = MSEloss + 0.0001 * regularization_loss

    if (t + 1) % 1 == 0:        # 每训练1个epoch，打印一次损失函数的值
        print('epoch [{}/{}], loss:{:.4f}'.format(t + 1, 15, loss.item()))  # 每训练1个epoch，打印一次损失函数的值

    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度

    if (t + 1) % 5 == 0:
        torch.save(model.state_dict(), "./pytorch_NormalizeAutoEncoder_model.pkl")  # 每5个epoch保存一次模型
        print("save model")
#----------------------------------------------------------------------------------------------------------------------#

