# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/30
#  @Author: MiJizong
#  @Content: MIMO——Pytorch
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: Null
# ----------------------开发者信息-----------------------------------------
# ----------------------   代码布局： -------------------------------------
# 1、导入相关的包
# 2、数据预处理
# 3、建立模型
# 4、模型显示与编译
# ----------------------   代码布局： --------------------------------------
# --------------------  1、 导入相关的包 -----------------------------------

import torch
import torch.nn as nn
import numpy as np
from numpy import random as rd
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块显卡
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
# --------------------  1、 导入相关的包 -----------------------------------
# --------------------  2、 数据预处理 -------------------------------------

samples_n = 3000
samples_dim_01 = 2
samples_dim_02 = 2
# 随机生成样本数据
x1 = rd.rand(samples_n, samples_dim_01)
x2 = rd.rand(samples_n, samples_dim_02)
y_1 = []
y_2 = []
# y_3 = []
for x11, x22 in zip(x1, x2):  # zip() 打包元素为元组，返回由这些元组组成的列表
    # zip 方法在 Python 2 和 Python 3 中的不同：在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。
    y_1.append(np.sum(x11) + np.sum(x22))
    y_2.append(np.max([np.max(x11), np.max(x22)]))
    # y_3.append(np.min([np.min(x11), np.min(x22)]))
y_1 = np.array(y_1)
y_1 = np.expand_dims(y_1, axis=1)  # 在1位置添加数据,，扩展数组的形状
y_2 = np.array(y_2)
y_2 = np.expand_dims(y_2, axis=1)
# y_3 = np.array(y_3)
# y_3 = np.expand_dims(y_3, axis=1)

x1, x2 = torch.Tensor(x1),torch.Tensor(x2)
y_1, y_2 = torch.Tensor(y_1), torch.Tensor(y_2)


# --------------------  2、 数据预处理 -------------------------------------

# --------------------  3、 模型建立 ---------------------------------------
class MIMO(nn.Module):
    def __init__(self):
        super(MIMO,self).__init__()
        # 全连接层
        self.dense_01 = nn.Linear(2,10)
        self.softmax1 = nn.Softmax()
        self.dense_11 = nn.Linear(10,10)
        self.softmax2 = nn.Softmax()

        self.dense_02 = nn.Linear(2,10)

        self.output_01 = nn.Linear(20,1)
        self.relu = nn.ReLU()
        self.output_02 = nn.Linear(20,1)
    def forward(self,input1,input2):
        output_01 = self.dense_01(input1)
        output_01 = self.softmax1(output_01)
        output_11 = self.dense_11(output_01)
        output_11 = self.softmax2(output_11)

        output_02 = self.dense_02(input2)

        # 加入合并
        merge = torch.cat((output_11,output_02),dim=1)
        output_01 = self.output_01(merge)
#        output_11 = self.output_01(output_01)
        output_02 = self.output_02(merge)   # 与output_011设置相同，直接使用
        return output_01,output_02

mimo = MIMO()
print(mimo)
# --------------------  3、 模型建立 ---------------------------------------

# --------------------  4、 模型显示与编译 ----------------------------------
optimizer = torch.optim.Adam(mimo.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()

loss_list = []  # 建立一个loss的列表，以保存每一次loss的数值
for t in range(3000):
        out1,out2 = mimo(x1,x2)
        loss1 = loss_func(out1, y_1)  # 计算损失
        loss2 = loss_func(out2, y_2)
        loss = (loss1+loss2)
        loss_list.append(loss)        # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()         # 梯度清零
        loss.backward()               # 反向传播
        optimizer.step()              # 参数更新

        if t % 50 == 0:
            print(f"第{t}轮   loss1为{loss1},   loss2为{loss2},   总训练损失为：{loss}")  # 打印损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
# --------------------  4、 模型显示与编译 ----------------------------------


