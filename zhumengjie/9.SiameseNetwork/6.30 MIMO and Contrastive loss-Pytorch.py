#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月30日
# 开发框架：Torch
# 开发内容：实现自定义contrastive loss及多输入多输出模型
#----------------------------------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入 torch， functional的包
# 2、MIMO模型搭建
# 3、自定义contrastive loss函数
# 4、定义优化器
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
#  --------------------------导入需要包 -------------------------------

#  -------------------------- 2、MIMO模型搭建 -------------------------------
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, (3,3), padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(24*24*16, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5)
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(256, 7),
            torch.nn.Softmax(),
        )

    def forward(self, input1, input2):
        out1 = self.layer1(input1)
        out2 = self.layer1(input2)
        out1 = self.layer2(out1)
        out2 = self.layer2(out2)
        out1 = self.layer3(out1)
        out2 = self.layer3(out2)
        return out1, out2

model = Net()
print(model)
#  -------------------------- MIMO模型搭建 -------------------------------

#  -------------------------- 3、自定义contrastive loss-------------------------------
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
#  -------------------------- 3、自定义contrastive loss-------------------------------

#  -------------------------- 4、定义优化器-------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = ContrastiveLoss()
#  -------------------------- 4、定义优化器-------------------------------