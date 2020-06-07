# -------------------------------------------------开发者信息----------------------------------------------------------#
# 开发者：高强
# 开发日期：2020年6月3日
# 开发框架：pytorch
# 温馨提示：服务器上跑
#----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------代码布局-----------------------------------------------------------#
# 1、读取手写体数据及与图像预处理
# 2、构建自编码器模型
# 3、训练过程可视化
# 4、保存模型

#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------读取手写体数据及与图像预处理------------------------------------------------#
import numpy as np
# 载入数据：本地
# path = 'F:\\Keras代码学习\\keras\\keras_datasets\\mnist.npz'
# 载入数据：服务器
path = 'mnist.npz'
f = np.load(path)
print(f.files) # 查看文件内容 ['x_test', 'x_train', 'y_train', 'y_test']
# 定义训练数据 60000个
x_train = f['x_train']
# 定义测试数据 10000个
x_test = f['x_test']
f.close()

# 数据预处理
# 数据格式进行转换
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
# 打印训练数据的维度
print(x_train.shape)  # (60000, 28, 28)
# 打印测试数据的维度
print(x_test.shape)   # (10000, 28, 28)

# 数据预处理
# 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.
#----------------------------------------------构建噪声数据集----------------------------------------------------------#
'''
关于：np.random.normal
loc：float  此概率分布的均值（对应着整个分布的中心centre）
scale：float 此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints   输出的shape，默认为None，只输出一个值
'''
noise_factor = 0.7
x_train_noise = x_train + noise_factor * np.random.normal(loc = 0.0,scale = 1.0,size = x_train.shape)
x_test_noise = x_test + noise_factor * np.random.normal(loc = 0.0,scale = 1.0,size = x_test.shape)

'''
np.clip()的三个参数:
第一个为数组,
使用第二个参数代替数组中的最小数(比0小的全部替代为0)
使用第三个参数代替数组中的最大数(比1大的全部替代为1)
'''
x_train_noise = np.clip(x_train_noise,0.,1.) # 控制训练集的数据在0-1之间
x_test_noise = np.clip(x_test_noise,0.,1.)   # 控制测试集的数据在0-1之间


#----------------------------------------------------------------------------------------------------------------------#

import torch
x_train_noise = torch.Tensor(x_train_noise)  # 转换为tenser
x_test_noise = torch.Tensor(x_test_noise)    # 转换为tenser
#----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------构建卷积自编码器模型------------------------------------------------------#


import torch.nn as nn
class myModel(nn.Module):
    def __init__(self):
        super(myModel, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=1,padding=1),  # 1*28*28 --> 32*28*28
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2)                                                    # 32*28*28 --> 32*14*14
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1),  # 32*14*14 --> 32*14*14
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2)                                                    # 32*14*14--> 32*7*7
        )


        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 32*7*7--> 32*7*7
            torch.nn.ReLU(),
            torch.nn.Upsample((14,14))        #  根据不同的输入类型制定的输出大小               # 32*7*7-> 32*14*14
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 32*14*14--> 32*14*14
            torch.nn.ReLU(),
            torch.nn.Upsample((28,28))                                                           # 32*28*28--> 32*28*28
        )


        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),  # 32*28*28-> 1*28*28
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # input[60000, 28, 28, 1]---->[6000,1,28,28] 交换位置
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        output = self.output(conv4)
        return output

model = myModel()


#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------训练过程可视化、保存模型-------------------------------------------------------#

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
Epoch = 15
## 开始训练 ##
for t in range(15):

    x = model(x_train_noise)          # 向前传播
    loss = loss_fn(x, x_train_noise)  # 计算损失

    if (t + 1) % 1 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(t + 1, 15, loss.item()))  # 每训练1个epoch，打印一次损失函数的值

    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度

    if (t + 1) % 5 == 0:
        torch.save(model.state_dict(), "./pytorch_NormalizeAutoEncoder_model.pkl")  # 每5个epoch保存一次模型
        print("save model")
#----------------------------------------------------------------------------------------------------------------------#







