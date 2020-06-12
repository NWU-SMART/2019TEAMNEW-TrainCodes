# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月12日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch.nn as nn
import torch
from torch.autograd import Variable
import os
from PIL import Image
import numpy as np
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、定义模型   --------------------------------
class TwoInOutputsNet(nn.Module):
  def __init__(self):
    super(TwoInputsNet, self).__init__()
    # input1 经过的卷积操作得到输出c
    self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
    self.conv11 = nn.Conv2d(64, 3, 3, 1, 1)

    # input2 经过的卷积操作得到输出f
    self.conv2 = nn.Conv2d(3, 64, 7, 1, 3)
    self.conv22 = nn.Conv2d(64, 3, 3, 1, 1)

    # c 和 f 相乘后得到 d，分别输入conv3，conv4 ，得到output1,output2
    self.conv3 = nn.Conv2d(3, 3, 5, 1, 2)
    self.conv4 = nn.Conv2d(3, 3, 7, 1, 3)

  def forward(self, input1, input2):
    c = self.conv1(input1)
    c = self.conv11(c)

    f = self.conv2(input2)
    f = self.conv22(f)

    d = torch.mul(c, f)

    out1 = self.conv3(d)
    out2 = self.conv4(d)
    return out1, out2
#  -------------------------- 2、定义模型   --------------------------------

#  -------------------------- 3、导入数据   --------------------------------
data_dir = "dataset/image"  # input1位置
label_dir ="dataset/labels"  # input2位置
def load_data(data_dir):
    datas = []
    labels = []
    for fname in os.listdir(data_dir): # 获取图片的文件名
        fpath = os.path.join(data_dir, fname) # 将获取的文件名与训练集路径拼接
        image = Image.open(fpath) # 打开图片
        data = np.array(image) / 255.0 #归一化处理
        datas.append(data) #将图片添加到datas数组

    for fname in os.listdir(label_dir): # 获取图片的文件名
        fpath = os.path.join(label_dir, fname) # 将获取的文件名与训练集路径拼接
        image = Image.open(fpath) # 打开图片
        data = np.array(image) / 255.0 #归一化处理
        labels.append(data) #将图片添加到labels数组

    # 转换为numpy数组
    datas = np.array(datas)
    labels = np.array(labels)
    return datas, labels

datas, labels = load_data(data_dir) # 调用load_data
(X, Y) = (datas, labels)# 获取数据

X = Variable(torch.from_numpy(X)).float()# 转为variable 数据类型
Y = Variable(torch.from_numpy(Y)).float()# 转为variable 数据类型

X = X.permute(0,3,2,1) #调整channel位置
Y = Y.permute(0,3,2,1) #调整channel位置
#  -------------------------- 3、导入数据   --------------------------------

#  -------------------------- 4、开始训练   --------------------------------
model = TwoInOutputsNet()#载入模型
loss_fn = nn.MSELoss() #损失函数
learning_rate = 1e-4 # 学习率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #SGD优化器

for epoch in range(10):
    x, y = model(X, Y)# 输入input1,input2，得到output1,output2
    loss = loss_fn(x, y)# 计算损失函数
    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播
    optimizer.step()#梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 10, loss.item()))
    #每训练1个epoch，打印一次损失函数的值
#  -------------------------- 4、开始训练   --------------------------------
