#---------------------------------------------------------开发者信息-----------------------------------------------------
#开发者：王园园
#开发日期：2020.6.10
#开发软件：pycharm
#开发项目：LSTM(长短期记忆网络)

#----------------------------------------------------------导包---------------------------------------------------------
import torch
import torchvision
import torchvision.datasets as dsets
import torch.utils.data as Data
from torch import nn
from torch.autograd import Variable

#-----------------------------------------------------------设置参数-----------------------------------------------------
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False
torch.manual_seed(1)
#-------------------------------------------------------------加载数据---------------------------------------------------
#训练数据集
train_data = dsets.MNIST(
    root='./mnist',          #mnist数据集路径
    train=True,               #是否训练
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
#测试数据集
test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
#批量加载数据
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
#数据归一化
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)/255
#测试数据的标签
test_y = test_data.test_labels

#-----------------------------------------------------------构建模型及训练-----------------------------------------------
#基于RNN的LSTM
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

rnn = RNN()   #实例化模型
print(rnn)

#优化器
optimizer = torch.optim.Adam(rnn.parameters(), lr = LR)
#交叉熵损失函数
loss_func = nn.CrossEntropyLoss()

#训练网络
for epoch in range(EPOCH):
    for step,(x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))
        b_y = Variable(y)

        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50 == 0:
            test_output = rnn(test_x.view(-1, 28, 28))
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y)/float(test_y.size(0))
            print('Epoch:', epoch, '|train loss:%.4f' %loss.data[0], '|test accuracy:%.2f' %accuracy)

#输出训练结果
test_output = rnn(test_x[:10].view(-1, 28, 28))
#预测结果
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')


