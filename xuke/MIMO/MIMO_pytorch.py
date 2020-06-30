# --------------         开发者信息--------------------------
# 开发者：徐珂
# 开发日期：2020.6.30
# software：pycharm
# 项目名称：MIMO（Ppytorch）
# --------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包  sss panda是一个可数据预处理的包
# 2、手写数据集导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测
# ----------------------   代码布局： ----------------------

#  --------------------- 1、导入需要包 ----------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.distributed as dist
import torch.utils.data as data_utils
#  --------------------- 1、导入需要包 ----------------------
# ---------------------------------2、处理数据-----------------------------
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
    y_3.append(np.min([np.min(x11), np.min(x22)]))
y_1 = np.array(y_1)
y_1 = np.expand_dims(y_1, axis=1)
y_2 = np.array(y_2)
y_2 = np.expand_dims(y_2, axis=1)
y_3 = np.array(y_3)
y_3 = np.expand_dims(y_3, axis=1)
# ---------------------------------2、处理数据-----------------------------

# -------------------------------3、模型构建-------------------------------
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

        out = torch.cat((out04, out14), dim=1)
        out1 = self.predict1(out)
        out2 = self.predict2(out)

        return out1, out2  # 多输出


net = Net(1, 20, 1)

x1 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 随便弄一个数据
y1 = x1.pow(3) + 0.1 * torch.randn(x1.size())

x2 = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y2 = x2.pow(3) + 0.1 * torch.randn(x2.size())

x1, y1 = (Variable(x1), Variable(y1))
x2, y2 = (Variable(x2), Variable(y2))

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
loss_func = torch.nn.MSELoss()
# ---------------------------------3、模型构建-----------------------------
# ---------------------------------4、训练---------------------------------
for t in range(5000):
    prediction1, prediction2 = net(x1, x2)
    loss1 = loss_func(prediction1, y1)
    loss2 = loss_func(prediction2, y2)
    loss = loss1 + loss2  # 重点！

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 100 == 0:
        print('Loss1 = %.4f' % loss1.data, 'Loss2 = %.4f' % loss2.data, )
# ---------------------------------4、训练---------------------------------