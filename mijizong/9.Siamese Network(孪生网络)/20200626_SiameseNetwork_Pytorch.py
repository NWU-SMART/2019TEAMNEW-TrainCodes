# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/26
#  @Author: MiJizong
#  @Content: SiameseNetwork——Pytorch
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
'''
孪生神经网络用于处理两个输入"比较类似"的情况。伪孪生神经网络适用于处理两个输入"有一定差别"的情况。
比如，我们要计算两个句子或者词汇的语义相似度，使用siamese network比较适合；如果验证标题与正文的
描述是否一致（标题和正文长度差别很大），或者文字是否描述了一幅图片（一个是图片，一个是文字），就
应该使用pseudo-siamese network。

三胞胎连体: 输入是三个，一个正例+两个负例，或者一个负例+两个正例，
Triplet在cifar, mnist的数据集上，效果超过了siamese network。
'''
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入需要包
# 2、函数功能区
# 3、主调区
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
from torchsummary import summary
#  -------------------------- 1、导入需要包 -------------------------------

# -----------------------------2、函数功能区(API)--------------------------
class FeatureNetwork(nn.Module):
    """生成特征提取网络"""
    """这是根据，MNIST数据调整的网络结构，下面注释掉的部分是，原始的Matchnet网络中feature network结构"""
    def __init__(self):
        super(FeatureNetwork,self).__init__()
        self.conv1 = nn.Sequential(
                                   # 网络第一层
                                   nn.Conv2d(1,24,kernel_size=3,padding=1),   # 1*28*28 -> 24*28*28
                                   nn.ReLU(),
                                   nn.MaxPool2d((3,3)),             # 24*28*28 -> 24*9*9

                                   # 网络第二层
                                   nn.Conv2d(24,64,kernel_size=3,padding=1),  # 24*9*9 -> 64*9*9
                                   nn.ReLU(),

                                   # 网络第三层
                                   nn.Conv2d(64,96,kernel_size=3,padding=1),  # 64*9*9 -> 96*9*9
                                   nn.ReLU(),

                                   # 网络第四层
                                   nn.Conv2d(96,96,kernel_size=3,padding=1),  # 96*9*9 -> 96*9*9
                                   nn.ReLU(),

                                   # 网络第五层
                                   nn.Flatten(),  # 96*9*9=7776
                                   nn.Linear(7776,512),
                                   nn.ReLU())
    def forward(self,x):
        x = self.conv1(x)
        return x

# 继承
class ClassiFilerNet(FeatureNetwork):  # add classifier Net
    """生成度量网络和决策网络，其实matchnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)"""

    def __init__(self):
        super(ClassiFilerNet,self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(1024,1024),
                                    nn.ReLU(),
                                    nn.Linear(1024,256),
                                    nn.ReLU(),
                                    nn.Linear(256,2),
                                    nn.Softmax())
    def forward(self,input1,input2):
        output1 = self.conv1(input1)
        output2 = self.conv1(input2)
        output = torch.cat((output1,output2),1)  # 拼接
        output = self.dense1(output)
        return output

# -----------------------------2、函数功能区(API)--------------------------

# ------------------------------3、主调区----------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matchnet = ClassiFilerNet()
matchnet = matchnet.to(device)
summary(matchnet,[(1, 28, 28), (1, 28, 28)])
# ------------------------------3、主调区----------------------------------