# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月1日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、定义模型   --------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=3, padding=1),  # 输入 1x28x28，经过卷积得到 24x10x10
            nn.ReLU(),# relu激活函数
            nn.nn.MaxPool2d(2, stride=2),  # 24x5x5

            nn.Conv2d(24, 64, kernel_size=3, stride=2, padding=1),  # 64x3x3
            nn.ReLU(),# relu激活函数
            nn.nn.MaxPool2d(2, stride=1),  # 64x2x2

            nn.Conv2d(64, 96, kernel_size=1, stride=1),  # 96x2x2
            nn.ReLU(),# relu激活函数

            nn.Conv2d(96, 96, kernel_size=1, stride=1),  # 96x2x2
            nn.ReLU()# relu激活函数
            )
        self.fc1 = nn.Sequential(
            nn.Linear(384, 10), #全连接层 
            nn.ReLU()# relu激活函数
            )
    def forward_once(self, x):
        output = self.cnn1(x)# 经过卷积层
        output = output.view(output.size()[0], -1)# 将向量展平
        output = self.fc1(output)#经过全连接层1
        return output
      
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)# 输入input1，得到output1 
        output2 = self.forward_once(input2)# 输入input2, 得到output2
        return output1, output2
#  -------------------------- 2、定义模型   --------------------------------

#  -------------------------- 3、定义对比损失函数   --------------------------------
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # 定义容忍度
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2) # 计算欧氏距离
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +  
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # 对比损失函数：[（1-Y）* 1/2 * (euclidean_distance)²] + [Y * 1/2 *{max(0,margin - euclidean_distance)}²]

        return loss_contrastive
#  -------------------------- 3、定义对比损失函数   --------------------------------      

