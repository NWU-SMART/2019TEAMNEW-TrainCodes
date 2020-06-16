#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月16日
# 开发框架：pytorch
# 开发内容：使用LSTM网络实现手写数字识别(pytorch)
#----------------------------------------------------------#
'''
代码在服务器中跑，电脑带不起来
CrossEntropyLoss(): 1、函数中包含独热编码，所以不使用独热编码。
                    2、输入img要修改为float()格式float32，否则跟weight不匹配报错，img = img.float()
                       输入label要修改为long()格式int64，否则跟交叉熵公式不匹配报错，label = label.long()
'''

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, os 的包
# 2、读取数据和数据处理
# 3、参数定义
# 4、built the LSTM model
# 5、模型训练
# 6、模型测试
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
from torch import nn
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
#  -------------------------- 导入需要包 -------------------------------

#  -------------------------- 2、读取数据和数据处理-------------------------------
# 数据集本地路径
path = 'mnist.npz'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
x_train = f['x_train']
y_train = f['y_train']
# 测试数据
x_test = f['x_test']
y_test = f['y_test']
f.close()

# 将图片信息转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 数据reshape和归一化
x_train = x_train.reshape(-1, 28, 28) / 255
x_test = x_test.reshape(-1, 28, 28) / 255

# label信息转换数据类型
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')

# 转换为tensor形式
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

# 打印数据shape
print('x_train.shape',x_train.shape)
print('y_train.shape',y_train.shape)
print('x_test.shape',x_test.shape)
print('y_test.shape',y_test.shape)


#  -------------------------- 读取数据和数据处理-------------------------------

#  -------------------------- 3、参数定义-------------------------------
EPOCH = 5               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
TIME_STEP = 28          # 时间步数 / 图片高度
INPUT_SIZE = 28         # 每步输入值 / 图片每行像素
LR = 0.01               # learning rate
#  -------------------------- 参数定义-------------------------------

#  -------------------------- 4、built the LSTM model-------------------------------
class lstm(nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.rnn = nn.LSTM(
        # Long-short term memory 普通RNN会可能产生梯度消失或者梯度爆炸无法回忆起久远记忆
            input_size=INPUT_SIZE,
            hidden_size=64,         # 隐藏层
            num_layers=1,           # 层数
            batch_first=True,       # （batch, 28, 28)
        )
        self.out = nn.Linear(64, 10) # 输出层

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的 r_out 输出
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :])
        return out

model = lstm()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)   # 优化器
loss_func = nn.CrossEntropyLoss()                       # 损失函数

#  -------------------------- built the LSTM model-------------------------------

#  -------------------------- 5、模型训练-------------------------------
print("-----------训练开始-----------")

for i in range(EPOCH):
    # 预测结果
    pred = model(x_train)
    # 计算损失
    print(pred.shape)
    loss = loss_func(pred, y_train.long())
    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 梯度更新
    optimizer.step()
    print(i, loss.item())

print("-----------训练结束-----------")
torch.save(model.state_dict(), "LSTM.pkl")  # 保存模型参数
# -------------------------------模型训练------------------------

#  -------------------------- 5、模型测试 -------------------------------
print("-----------测试开始-----------")

model.load_state_dict(torch.load('LSTM.pkl')) # 加载训练好的模型参数

for i in range(EPOCH):
    # 预测结果
    pred = model(x_test)
    # 计算损失
    loss = loss_func(pred, y_test.long())
    # 打印迭代次数和损失
    print(i, loss.item())

    # 打印图片显示decoder效果
    pred = pred.detach().numpy()
    plt.imshow(pred[i].reshape(28, 28))
    plt.gray()  # 显示灰度图像
    plt.show()

print("-----------测试结束-----------")
# -------------------------------模型测试------------------------