# ----------------开发者信息--------------------------------
# 开发者：姜媛
# 开发日期：2020年6月17日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------


#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from keras.layers import Dense, Input
from keras.models import Model
#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、读取数据和数据处理-------------------------------
# 训练数据
DOWNLOAD_MNIST = False
train_data = dsets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
# 训练数据加载
train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)

# 测试数据
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)/255.   # (2000, 28, 28) 归一化
test_y = test_data.test_labels.numpy()                      # 转换为 numpy array
#  -------------------------- 2、读取数据和数据处理-------------------------------


#  -------------------------- 3、参数定义-------------------------------
EPOCH = 5               # train the training data n times, to save time, we just train 1 epoch
TIME_STEP = 28          # 时间步数 / 图片高度
INPUT_SIZE = 28         # 每步输入值 / 图片每行像素
LR = 0.01
#  --------------------------3、 参数定义-------------------------------


#  -------------------------- 4、built the LSTM model-------------------------------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,         # 隐藏层通道
            num_layers=1,           # LSTM 层数
            batch_first=True,       # （batch, time_step, input_size)
        )                           # Long-short term memory 普通RNN会可能产生梯度消失或者梯度爆炸无法回忆起久远记忆
        self.out = nn.Linear(64, 10) # 输出层

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out


model = LSTMModel()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # 优化器
loss_func = nn.CrossEntropyLoss()                         # 损失函数，label不用独热编码
#  -------------------------- 4、built the LSTM model-------------------------------


#  -------------------------- 5、模型训练和测试-------------------------------
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):    # batch 数据
        b_x = b_x.view(-1, 28, 28)                      # reshape x 为 (batch, time_step, input_size)
        output = model(b_x)                             # 输出
        loss = loss_func(output, b_y)                   # 交叉熵损失
        optimizer.zero_grad()                           # 梯度清除
        loss.backward()                                 # 方向传播
        optimizer.step()                                # 参数更新

        if step % 50 == 0:
            test_output = model(test_x)                 # (samples, time_step, input_size)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

# 从测试中打印10个预测数字
test_output = model(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()  # 预测结果
print(pred_y, 'prediction number')                  # 打印预测数字
print(test_y[:10], 'real number')                   # 打印真实数字
# -------------------------------5、模型训练和测试------------------------
