#-------------------------------------------------------开发者信息----------------------------------------
#开发人：王园园
#开发日期：2020.5.30
#开发软件：pycharm
#开发项目：正则自编码器（pytorch）

#----------------------------------------------------代码布局--------------------------------------------
#1、导包
#2、读取手写体数据及与图像预处理
#3、构建自编码器模型
#4、模型可视化
#5、训练
#6、查看解码效果
#7、训练过程可视化

#----------------------------------------------------导包------------------------------------------------
import os
from tkinter import Variable
import torch
from cv2 import datasets
from matplotlib import transforms
from networkx.drawing.tests.test_pylab import plt
from torch import nn, optim
from torch.utils.data import DataLoader

#----------------------------------------------------加载数据---------------------------------------------
def get_data():
    # 将像素点转换到[-1, 1]之间，使得输入变成一个比较对称分布，训练容易收敛
    data_tf = transforms.Compose([transforms.Totensor(), transforms.Normalize([0.5, 0.5])])
    train_dataset = datasets.MNIST(root='D:/keras_datasets/mnist.npz', train=True, transform=data_tf, download=True)
    train_loder = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    return train_loder

#-----------------------------------------------------编码解码---------------------------------------------
class normalizeEncoder(nn.Module):
    def __init__(self):
        super(normalizeEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 1*784),
                                     nn.ReLU(True),
                                     nn.Linear(784, 32),
                                     nn.Sigmoid(True))
        self.decoder = nn.Sequential(nn.Linear(32, 784),
                                     nn.Sigmoid(True))
    def forward(self, x):
        encode= self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

#---------------------------------------------------------训练---------------------------------------------
def to_img(data):
    pass

if __name__ == "__name__":
    batch_size = 128
    lr = 1e-2
    weight_decay = 1e-5   #权重衰减
    epoches = 5
    model = normalizeEncoder()     #调用模型
    train_data = get_data()    #取数据
    criterion = nn.MSELoss     #损失
    optimizier = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(epoches):         #循环epoch，设置学习率
        if epoch in [epoches * 0.25, epoches * 0.5]:
            for param_group in optimizier.param_groups:
                param_group['lr'] *= 0.1
        for img, _ in train_data:
            img = img.view(img.size(0), -1)
            img = Variable(img.cuda())
            # forward
            _, output = model(img)  # 把数据放入模型
            loss = criterion(output, img)
            # backward
            optimizier.zero_grad()
            loss.backward()
            optimizier.step()
        print('epoch=', epoch, loss.data.float())
        for param_group in optimizier.param_groups:
            print(param_group['lr'])
        if (epoch + 1) % 5 == 0:
            pic = to_img(output.cpu().data)
            if not os.path.exists('./simple_autoencoder'):  # 判断该路径是否存在
                os.mkdir(pic, './simple_autoencoder/image_{}.png'.format(epoch + 1))  # 如果不存在则将处理后的数据放入该路径
    code = Variable(torch.FloatTensor([1.19, -3.36, 2.06]).cuda())
    decode = model.decoder(code)  # 解码
    decode_img = to_img(decode).squeeze()
    decode_img = decode_img.data.cpu().numpy() * 255
    plt.imshow(decode_img.astype('uint8'), cmap='gray')
    plt.show()