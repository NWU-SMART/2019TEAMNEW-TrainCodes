# -*- coding: utf-8 -*-
# @Time: 2020/6/18 20:49
# @Author: wangshengkang

import torch
import torch.nn as nn
from torchsummary import summary


# 孪生网络，相同的网络部分
class basenet(nn.Module):
    def __init__(self):
        super(basenet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),  # 24*28*28
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3)),  # 24*9*9

            nn.Conv2d(24, 64, 3, padding=1),  # 64*9*9
            nn.ReLU(),

            nn.Conv2d(64, 96, 3, padding=1),  # 96*9*9
            nn.ReLU(),

            nn.Conv2d(96, 96, 3, padding=1),  # 96*9*9
            nn.ReLU(),
            nn.Flatten(),  # 7776
            nn.Linear(7776, 512),  # 512
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn1(x)
        return x


# 继承basenet，合并以后的网络
class SiameseNetwork(basenet):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 1024),  # 1024
            nn.ReLU(),
            nn.Linear(1024, 1024),  # 1024
            nn.ReLU(),
            nn.Linear(1024, 2),  # 2
            nn.Softmax()
        )

    def forward(self, input1, input2):
        output1 = self.cnn1(input1)  # 相同的部分1
        output2 = self.cnn1(input2)  # 相同的部分2
        output3 = torch.cat((output1, output2), 1)  # 两个分支的特征拼接起来
        output3 = self.fc1(output3)  # 合并后的网络
        return output3


net = SiameseNetwork()
print(net)
summary(net, [(1, 28, 28), (1, 28, 28)])  # 打印模型结构，给定输入尺寸
