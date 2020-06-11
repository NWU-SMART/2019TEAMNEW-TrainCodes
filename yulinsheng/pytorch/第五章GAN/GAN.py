# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.9
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
# 4.超参数建立
# 4.生成器和判别器模型建立
# 5.训练
# 6.训练结果可视化
# 7.查看效果
# /------------------ 代码布局 --------------------*/


# /------------------ 导入需要的包--------------------*/
import numpy as np
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
# /------------------ 读取数据--------------------*/
# /------------------ 数据预处理 --------------------*/

# 归一化操作
x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
# /------------------ 数据预处理 --------------------*/

# /------------------ 生成器和判别器模型建立 --------------------*/

import torch
from torch.nn import Linear,LeakyReLU,BatchNorm1d,Sequential,ReLU,Dropout,Sigmoid
# 生成器
# 通过输入噪声200维度，生成1*28*28的图像(784)
class generation(torch.nn.Module):
    def __init__(self):
        super(generation,self).__init__()
        self.generator = Sequential(
            # 200->256
            Linear(200,256),
            BatchNorm1d(256),
            ReLU(True),
            # 256->28*28*1
            Linear(256,784),
            ReLU(True)
        )
    def forward(self,x):
        img_generator = self.generator(x)
        return img_generator
# 判别器模型
# 输入784，输出2维
class discriminator(torch.nn.Module):
    def __init__(self):
        super(discriminator,self).__init__()
        self.dicrimination = Sequential(
            Linear(784,256),
            LeakyReLU(0.2),
            Dropout(0.25),
            Linear(256,64),
            LeakyReLU(0.2),
            Dropout(0.25),
            Linear(64,1),
            Sigmoid()
        )
    def forward(self, x):
        x = torch.FloatTensor(x)
        img_dicrimination = self.dicrimination(x)
        return img_dicrimination
# 创建对象
D = discriminator()
G = generation()
# 将模型输入到cuda中
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()
# 定义损失函数
criterion = torch.nn.BCELoss()
d_optim = torch.optim.Adam(D.parameters(),lr=1e-5)
g_optim = torch.optim.Adam(G.parameters(),lr=1e-4)

# /------------------ 生成器和判别器模型建立 --------------------*/
# 训练报错
from torch.autograd import Variable

import torch
import matplotlib.pyplot as plt
import numpy as np
for epoch in range(200):
    for i in range(10000):
        img = x_train[i,:]
        img = torch.tensor(img)
        img = Variable(img).cuda()
        real_label = Variable(torch.ones(1)).cuda()  # 定义真实的图片label为1
        fake_label = Variable(torch.zeros(1)).cuda()  # 定义假的图片的label为0
        real_out = D(img)
        d_loss_true = criterion(real_out,real_label)
        noise = np.random.uniform(0,1,size=[10000,200])
        noise = torch.tensor(noise)
        noise = Variable(noise).cuda()  # 随机生成一些噪声
        fake_img = G(noise).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
        fake_out = D(fake_img)  # 判别器判断假的图片，
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        # 损失函数和优化
        d_loss = d_loss_true + d_loss_fake  # 损失包括判真损失和判假损失
        d_optim.zero_grad()  # 在反向传播之前，先将梯度归0
        d_loss.backward()  # 将误差反向传播
        d_optim.step()  # 更新参数
        z = np.random.uniform(0,1,size=[10000,200])
        z = torch.tensor(z)
        z = Variable(z).cuda()  # 得到随机噪声
        fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
        output = D(fake_img)  # 经过判别器得到的结果
        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
        # bp and optimize
        g_optim.zero_grad()  # 梯度归0
        g_loss.backward()  # 进行反向传播
        g_optim.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
        plt.plot(g_loss.data.item())
        plt.plot(d_loss.data.item())
        plt.savefig('loss.png')
    plt.show()


