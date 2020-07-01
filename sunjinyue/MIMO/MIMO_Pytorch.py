# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月30日
# 修改日期：
# 修改人：
# 修改内容：
from numpy import random as rd
import torch.nn as nn
import torch
import numpy as np

samples_n = 3000
samples_dim_01 = 2
samples_dim_02 = 2
# 样本数据
x1 = rd.rand(samples_n, samples_dim_01)
x2 = rd.rand(samples_n, samples_dim_02)
y_1 = []
y_2 = []
y_3 = []
for x11, x22 in zip(x1, x2):
  y_1.append(np.sum(x11) + np.sum(x22))
  y_2.append(np.max([np.max(x11), np.max(x22)]))
y_1 = np.array(y_1)
y_1 = np.expand_dims(y_1, axis=1)
y_2 = np.array(y_2)
y_2 = np.expand_dims(y_2, axis=1)

x1, y_1 = torch.Tensor(x1), torch.Tensor(y_1)
x2, y_2 = torch.Tensor(x2), torch.Tensor(y_2)

print(x1.shape)


class Net(nn.Module):
  def __init__(self, n_input, n_hidden, n_output):
    super(Net, self).__init__()
    self.hidden1 = nn.Linear(n_input, n_hidden)
    self.hidden2 = nn.Linear(n_hidden, n_hidden)

    self.predict1 = nn.Linear(n_hidden * 2, n_output)
    self.predict2 = nn.Linear(n_hidden * 2, n_output)

  def forward(self, input1, input2):  # 多输入
    out01 = self.hidden1(input1)
    out02 = torch.relu(out01)
    out03 = self.hidden2(out02)
    out04 = torch.sigmoid(out03)

    out11 = self.hidden1(input2)
    out12 = torch.relu(out11)
    out13 = self.hidden2(out12)
    out14 = torch.sigmoid(out13)

    out = torch.cat((out04, out14), dim=1)  # 模型层拼合

    out1 = self.predict1(out)
    out2 = self.predict2(out)

    return out1, out2  # 多输出

net = Net(2, 20, 1)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()

for t in range(5000):
    prediction1, prediction2 = net(x1, x2)
    loss1 = loss_func(prediction1, y_1)
    loss2 = loss_func(prediction2, y_2)
    loss = loss1 + loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 100 == 0:
       print('Loss1 = %.4f' % loss1.data,'Loss2 = %.4f' % loss2.data,)
