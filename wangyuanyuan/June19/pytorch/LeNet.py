#------------------------------------------------------开发者信息--------------------------------------------------------
#开发者：王园园
#开发日期：2020.6.19
#开发软件：pycharm
#开发项目：LeNet[MNIST数据集]（pytorch）
#MNIST数据集下载地址：http://yann.lecun.com/exdb/mnist/

#----------------------------------------------------------导包---------------------------------------------------------
import torch
import torchvision
from torch import nn, optim

#-----------------------------------------------------------搭建模型-----------------------------------------------------
#若检测到GPU环境则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#定义网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(                       #input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),                     #6个卷积核，卷积核大小为5*5，padding=2, 图片大小变为28+2*2=32（两边各加2列0），保证输入输出尺寸相同
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)         #input_size=(6*28*28), output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),                          #16个卷积核，卷积核的大小为5*5，input_size(6*14*14), output_size=16*10*10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)         #input_size=(16*10*10), output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(                          #全连接层
            nn.Linear(16*5*5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(                          #全连接层
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)                       #全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)                           #全连接层均使用的nn.Linear()线性结构，输入输出维度均为一维，故需要把数据拉为一维
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

#--------------------------------------------------------加载数据--------------------------------------------------------
#加载数据
transform = torchvision.transforms.ToTensor()        #定义数据预处理方式：转换PIL.Image成torch.FloatTensor
train_data = torchvision.datasets.MNIST(root='D:\SoftWare\Pycharm\MNIST',         #数据目录
                                        train = True,                                        #是否为训练集
                                        transform=transform,                                 #加载数据预处理
                                        download=True)                                      #是否下载，下载偏慢
test_data = torchvision.datasets.MNIST(root='D:\SoftWare\Pycharm\MNIST',
                                        train = False,
                                       transform=transform,
                                       download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)    #数据加载器：组合数据集核采样器
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False)

#定义loss
net = LeNet().to(device)                  #实例化网络，有GPU则将网络放入GPU加速
loss_fuc = nn.CrossEntropyLoss()          #多分类问题，选择交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)   #选择SGD，学习率取0.001

#---------------------------------------------------------训练----------------------------------------------------------
EPOCH = 8                    #训练总轮数
for epoch in range(EPOCH):
    sum_loss = 0
    #数据读取
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)       #有GPU则将数据置入GPU加速
        optimizer.zero_grad()                                       ##梯度清零
        #传递损失+更新参数
        output = net(inputs)
        loss = loss_fuc(output, labels)
        loss.backward()
        optimizer.step()

        #每训练100个batch打印一次平均loss
        sum_loss += loss.item()
        if i % 100 == 99:
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch+1, i+1, sum_loss/100))
            sum_loss = 0.0

    correct = 0
    total = 0

    for data in test_loader:
        test_inputs, labels=data
        test_inputs, labels = test_inputs.to(device), labels.to(device)
        outputs_test = net(test_inputs)
        _, predicted = torch.max(outputs_test.data, 1)                  #输出得分最高的类
        total += labels.size(0)                                         #统计50个batch 图片的总个数
        correct += (predicted == labels).sum()                          #统计50个batch 正确分类的个数

    print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct/total)))

