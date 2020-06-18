# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/17
#  @Author: MiJizong
#  @Content: 使用LSTM在手写体数据集上训练——Pytorch
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相应的包
# 2、读取手写体数据及与图像预处理
# 3、构建LSTM模型
# 4、模型训练与输出
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块显卡
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
# 以上两句命令如果不添加汇报下列错误：
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
# That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do
# is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static
# linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you
# can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute,
# but that may cause crashes or silently produce incorrect results. For more information, please see
# http://www.intel.com/software/products/support/.
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
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



# 数据预处理
#  归一化
x_train = x_train.reshape(-1, 28, 28) / 255  # -1代表未知
x_test = x_test.reshape(-1, 28, 28) / 255

# 将标签信息转化为int类型
y_train = y_train.astype('int64')
y_test = y_test.astype('int64')

# 将数据转换为Tensor格式
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)


input_size = 28
hidden_layer = 64

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、构建LSTM模型 ------------------------------------

class Lstm(nn.Module):
    def __init__(self):
        super(Lstm,self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_layer,  # 隐藏层状态的维数,就是LSTM在运行时里面的维度。
            num_layers=1,              # LSTM 堆叠的层数
            batch_first=True)          # True表示输入输出的第一维为batch_size
        self.out = nn.Linear(64,10)    # 定义输出层

    def forward(self,x):
        r_out,(h_n,h_c) = self.lstm(x,None)
        return self.out(r_out[:,-1,:])

lstm1 = Lstm()
print(lstm1)

#  --------------------- 3、构建LSTM模型 ------------------------------------


#  ----------------------- 4、模型训练与输出 ------------------------------
'''# 以下三行可以调用GPU加速训练，也就是在模型，x_train，y_train后面加上cuda()'''
lstm1 = lstm1.cuda()
x_train = x_train.cuda()
x_test = x_test.cuda()

loss_func = torch.nn.CrossEntropyLoss()                                   # 损失函数
optimizer = torch.optim.Adam(lstm1.parameters(),lr=1e-4)  # Adam优化器

#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
torch_dataset = Data.TensorDataset(x_train, x_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=64, shuffle=True)

loss_list = []  # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (x, y) in enumerate(loader):
        b_x = Variable(x.view(-1,28,28))
        b_y = Variable(y)
        train_prediction = lstm1(b_x)     # 一轮训练
        loss = loss_func(train_prediction, b_y)  # 计算损失
        loss_list.append(loss)       # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()        # 梯度清零
        loss.backward()              # 反向传播
        optimizer.step()             # 参数更新
        running_loss += loss.item()  # 损失叠加
    else:
        print(f"第{t}代训练损失为：{running_loss/len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
#  -------------------------- 4、模型训练与输出 -------------------------------