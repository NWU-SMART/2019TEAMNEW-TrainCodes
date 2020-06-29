# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月29日
# 内容:Siamese Network(不共享参数）
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 torch的包
# 2、函数功能区
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import torch
import torch.nn as nn
from torchsummary import summary
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、函数功能区 -----------------------------------
class Simesenetwork(nn.Module):
   def __init__(self):
       super(Simesenetwork,self).__init__()
       self.conv1=torch.nn.Sequential(
           torch.nn.Conv2d(in_channels=1,out_channels=24,kernel_size=(3,3),stride=1,padding=1),#1*28*28-24*28*28
           torch.nn.ReLU(),
           torch.nn.MaxPool2d(2,stride=2),#24*28*28-24*14*14
           torch.nn.Conv2d(in_channels=24,out_channels=64,kernel_size=(3,3),stride=1,padding=1),#24*14*14-64*14*14
           torch.nn.ReLU(),
           torch.nn.Conv2d(in_channels=64,out_channels=96,kernel_size=(3,3),stride=1,padding=1),#64*14*14-96*14*14
           torch. nn.ReLU(),
           torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=1, padding=1),
           torch. nn.ReLU(),
           torch. nn.Flatten(),
           torch.nn.Linear(18816, 512),
           torch.nn.ReLU()
       )
       self.dense1 = torch.nn.Sequential(
           torch.nn.Linear(1024, 1024),
           torch. nn.ReLU(),
           torch. nn.Linear(1024, 512),
           torch.nn.ReLU(),
           torch.nn.Linear(512, 2),
           torch.nn.Softmax()
       )
   def forward(self,input1,input2):
       output1=self.conv1(input1)
       output2=self.conv1(input2)
       output=torch.cat((output1,output2),1)
       output = self.dense1(output)
       return output


model=Simesenetwork()
summary(model,[(1,28,28),(1,28,28)])
