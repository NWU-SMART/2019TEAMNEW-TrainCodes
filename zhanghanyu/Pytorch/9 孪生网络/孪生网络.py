# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月26日
# 内容：孪生网络
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#
#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
# ---------------------函数功能区-------------------------
'''
    keras:padding默认为vaild,same的时候就会自动填充0.
    pytorch:padding默认为0,1的时候就会自动填充0.
'''
class FeatureNetwork(nn.Module):
    def __init__(self):
        super(FeatureNetwork, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,24,kernel_size=3,stride=1,padding=1),#(1*28*28)--(24*28*28)
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(24,64,3,1,1),
            nn.ReLU()
        )
        self.layer3=nn.Sequential(
            nn.Conv2d(64,96,3,1,0),
            nn.ReLU()
        )
        self.layer4=nn.Sequential(
            nn.Conv2d(96,96,3,1,0),
            nn.ReLU()
        )
        self.layer5=nn.Sequential(
            nn.Linear(96,512),
            nn.ReLU()

        )
    def forward1(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # 拉平，作用相当于Flatten
        x = self.layer5(x)
        return x
    def forward(self,input1,input2):
        output1 = self.forward1(input1)
        output2 = self.forward1(input2)
        return output1,output2

model = FeatureNetwork()
print(model)

# 打印模型模型参数总量
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
