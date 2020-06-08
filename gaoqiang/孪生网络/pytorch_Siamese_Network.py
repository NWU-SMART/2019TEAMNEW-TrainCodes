# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.08
# 开发框架：pytorch
# 温馨提示：
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局------------------------------------------------------------#
# 1、建立模型
# 2、保存模型与模型可视化
#----------------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,out_channels=24,kernel_size=3,stride=1,padding=1),  # 1*28*28 --> 24*28*28
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2),                                                  # 24*28*28 --> 24*14*14

            torch.nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, stride=1, padding=1),# 24*14*14 --> 64*14*14
            torch.nn.ReLU(),

            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1), # 64*14*14--> 96*14*14
            torch.nn.ReLU()
        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(18816, 512),
            torch.nn.ReLU()
        )

    def forward_once(self,x):
        output = self.cnn1(x)
        output = output.view(output.size(0), -1)  # 拉平，作用相当于Flatten
        output = self.fc1(output)
        return output

    def forward(self, input1,input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1,output2

model = SiameseNetwork()
print(model)






















