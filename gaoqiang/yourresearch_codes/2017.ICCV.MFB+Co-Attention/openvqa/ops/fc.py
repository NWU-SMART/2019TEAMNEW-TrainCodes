# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import torch.nn as nn
import torch

# 一层全连接层
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)       # 一层全连接

        if use_relu:                                     # 判断是否使用relu激活
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:                                # 如果dropput_r大于0，则使用dropout
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):                                # 前向传播
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

# 定义MLP层
class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)                               # 第二层全连接

    def forward(self, x):                                                          # 前向传播
        return self.linear(self.fc(x))
