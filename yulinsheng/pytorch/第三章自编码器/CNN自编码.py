# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.3
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/
# /------------------ 代码布局 --------------------*/
# 1.导入需要的包
# 2.读取图片数据
# 3.图片预处理
# 4.构建模型自编码器
# 5.训练
# 6.训练结果可视化
# 7.查看效果
# /------------------ 代码布局 --------------------*/


# /------------------ 导入需要的包--------------------*/
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像
# /------------------ 导入需要的包--------------------*/


# /------------------ 读取数据--------------------*/
# 数据路径
path = 'mnist.npz'
data = np.load(path)
# ['x_test', 'x_train', 'y_train', 'y_test']
# print(data.files)
# 读取数据
x_train = data['x_train']#(60000, 28, 28)
x_test = data['x_test']#(10000, 28, 28)
data.close()
# 归一化操作
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

import torch
x_train = torch.tensor(x_train).cuda()
x_test = torch.tensor(x_test).cuda()

# /------------------ 读取数据--------------------*/

# /------------------ 自编码器模型的建立--------------------*/

from torch.nn import Conv2d,MaxPool2d,ReLU,Sigmoid,Upsample,Sequential
import torch
# 输入的28*28*1
class autoencoder(torch.nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        # encoder
        self.conv1 = Sequential(
            Conv2d(in_channels=1,out_channels=4,padding=1,stride=1,kernel_size=3),
            # (28,28,4)
            ReLU(),
            MaxPool2d(kernel_size=2,dilation=1,padding=0,stride=2)
        #     (14,14,4)
        )#（28,28,1）->(28,28,4)->(14,14,4)
        self.conv2 = Sequential(
            Conv2d(in_channels=4,out_channels=8,padding=1,stride=1,kernel_size=3),
            # (14,14,8)
            ReLU(),
            MaxPool2d(kernel_size=2,dilation=1,padding=0,stride=2)
            # (7,7,8)
        )#（14,14,4）->(14,14,8)->(7,7,8)
        self.conv3 = Sequential(
            Conv2d(in_channels=8,out_channels=16,padding=1,stride=1,kernel_size=3),
            # (7,7,16)
            ReLU(),
            MaxPool2d(kernel_size=2,dilation=1,padding=0,stride=2)
        #     (4,4,16)
        )#(7,7,8)->(7,7,16)->(4,4,16)
        # decoder
        self.conv4 = Sequential(
            Conv2d(in_channels=16,out_channels=8,padding=1,stride=1,kernel_size=3),
            # (4,4,8)
            ReLU(),
            Upsample((8,8))
        #     (8,8,8)
        )
        self.conv5 = Sequential(
            Conv2d(in_channels=8,out_channels=4,padding=1,stride=1,kernel_size=3),
            # (8,8,4)
            ReLU(),
            Upsample((14,14))
        #     (14,14,4)
        )
        self.out= Sequential(
            Conv2d(in_channels=4,out_channels=1,padding=1,stride=1,kernel_size=3),
            # (14,14,1)
            Sigmoid(),
            Upsample((28,28))
        #     (28,28,1)
        )
    def forward(self,x):
        x = x.permute(0,3,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        decoder = self.out(x)
        return decoder
model = autoencoder().cuda()
import torch
loss_fn = torch.nn.BCELoss()
optimize = torch.optim.Adam(model.parameters(),lr=1e-3)

for i  in range(5):
    result = model(x_train)
    loss = loss_fn(result,x_train).cpu()
    optimize.zero_grad()
    loss.backward()
    optimize.step()
    print(i,loss.item())


# 查看decoder效果
torch.save(model,'cnn编码.h5')
model_new = torch.load('cnn编码.h5').cuda()
import matplotlib.pyplot as plt
# 打印图片显示decoder效果
new = model_new(x_test).cpu()
new = new.detach().numpy()
x_test = x_test.cpu()
n = 5
plt.figure(figsize=(20,8))#确定显示图片大小
for i in range(n):
    img = plt.subplot(2,5,i+1)
    plt.imshow(new[i].reshape(28,28))
    img = plt.subplot(2, 5, i + 6)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()#显示灰度图像
    plt.savefig('jieguo.jpg')
plt.show()

