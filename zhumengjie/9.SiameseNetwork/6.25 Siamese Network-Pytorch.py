#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月25日
# 开发框架：Pytorch
# 开发内容：孪生网络（共享参数）
#----------------------------------------------------------#

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
#  --------------------------导入需要包 -------------------------------
'''
    keras:padding默认为vaild,same的时候就会自动填充0.
    pytorch:padding默认为0,1的时候就会自动填充0.
'''

class SimeseNet(torch.nn.Module):
    def __init__(self):
        super(SimeseNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(3, 3), stride=1, padding=1),# 1*28*28->24*28*28
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3)),  # 24*28*28->24*7*7
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(3, 3), stride=1, padding=1), # 24*7*7->64*7*7
                nn.ReLU(),
            )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), stride=1, padding=0), # 64*7*7->96*7*7
                nn.ReLU(),
            )
        self.layer4 = nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=1, padding=0), # 96*7*7->96*7*7
                nn.ReLU(),
            )
        self.layer5 = nn.Sequential(
                nn.Linear(in_features=96*7*7, out_features=512),
                nn.ReLU(),
            )
    def forward_once(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # 拉平，作用相当于Flatten
        x = self.layer5(x)
        return x
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

model = SimeseNet()
print(model)

# 打印模型模型参数总量
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

