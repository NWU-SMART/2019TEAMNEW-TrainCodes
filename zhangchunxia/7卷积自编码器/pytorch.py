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
import torch
import torch.nn as nn
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
X_train = torch.Tensor(X_train)  # 转换为tenser
X_test = torch.Tensor(X_test)
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建自编码器模型 ------------------------------
class autoencoder(nn.Module):
    def __init__(self, ):
        super(autoencoder,self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,3,stride=1,padding=1),  # 1*28*28 --> 16*28*28
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2),#16*28*28 --> 16*14*14
            torch.nn.Conv2d(16, 8, 3, stride=1, padding=1),  # 16*14*14 --> 8*14*14
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),# 8*14*14--> 8*7*7
            torch.nn.Conv2d(8, 8, 3, stride=1, padding=1),# 8*7*7--> 8*7*7
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2)# 8*7*7--> 8*4*4
    )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, 3, stride=1, padding=1),#8*4*4--> 8*4*4
            torch.nn.ReLU(),
            torch.nn.Upsample((8, 8)),# 8*4*4--> 8*8*8
            torch.nn.Conv2d(8, 8, 3, stride=1, padding=1), # 8*8*8-> 8*8*8
            torch.nn.ReLU(),
            torch.nn.Upsample((16, 16)),# 8*8*8> 8*16*16
            torch.nn.Conv2d(8, 16, 3, stride=1, padding=0),#  8*16*16-> 16*14*14
            torch.nn.ReLU(),
            torch.nn.Upsample((16, 16)),#16*14*14-> 16*28*28
            torch.nn.Conv2d(16, 1, 3, stride=1, padding=1),#16*28*28->1*28*28
            torch.nn.Sigmoid()
        )
    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return(decode)
model = autoencoder()
#  ---------------------  3、构建自编码器模型 ------------------------------
#  ---------------------- 4、模型训练 ----------------------------------------
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
for i in range(5):

    X_P = model(X_train)          # 向前传播
    loss = loss_fn(X_P, X_train)  # 计算损失

    if (i + 1) % 1 == 0:        # 每训练1个epoch，打印一次损失函数的值
        print(loss.item())

    if (i + 1) % 5 == 0:
        torch.save(model.state_dict(), "./pytorch_CNNAutoEncoder_model.pkl")  # 每5个epoch保存一次模型
        print("save model")

    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度
#  ---------------------- 4、模型训练 ----------------------------------------
