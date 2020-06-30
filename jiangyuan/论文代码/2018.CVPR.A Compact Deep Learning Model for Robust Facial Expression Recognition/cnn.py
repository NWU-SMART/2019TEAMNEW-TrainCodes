import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# 定义CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),  # 输入3通道，输出16通道，卷积核为5*5，步长为1
            nn.PReLU(),  # PReLU层（带参数的ReLU层）
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2),
            # 输入16通道，输出16通道，卷积核为5*5，步长为1
            nn.PReLU(),  # PReLU层（带参数的ReLU层）
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，窗口尺寸2*2，步长为2
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            # 输入16通道，输出32通道，卷积核为5*5，步长为1
            nn.PReLU(),  # PReLU层（带参数的ReLU层）
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            # 输入32通道，输出32通道，卷积核为5*5，步长为1
            nn.PReLU(),  # PReLU层（带参数的ReLU层）
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层，窗口尺寸2*2，步长为2
        )
        # 两个全连接层
        self.fc1 = nn.Linear(32 * 24 * 24, 32 * 24 * 24)  # 全连接函数1为线性函数：y = Wx + b，并将32 * 24 * 24个节点连接到32 * 24 * 24个节点上
        self.fc2 = nn.Linear(32 * 24 * 24, 7)  # 全连接函数2为线性函数：y = Wx + b，并将32 * 24 * 24个节点连接到7个节点上

    # 定义前向传播方法
    def forward(self, x):
        x = self.conv1(x)  # 输入x经过卷积conv1之后，更新到x
        x = self.conv2(x)  # 输入x经过卷积conv2之后，更新到x
        x = x.view(x.size(0), -1)  # 将前面多维度张量展平成一维送入分类器
        x = F.dropout(x, p=0.6, training=self.training)  # 输入x经过训练后进行dropout，概率设为0.6，然后更新x
        x = self.fc1(x)  # 输入x经过全连接1，然后更新x
        x = F.dropout(x, p=0.6, training=self.training)  # 输入x经过训练后进行dropout，概率设为0.6，然后更新x
        x = self.fc2(x)  # 输入x经过全连接2，然后更新x
        return x
