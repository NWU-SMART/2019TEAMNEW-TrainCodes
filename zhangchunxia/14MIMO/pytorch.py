# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月1日
# 内容:MIMO
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、函数功能区
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import torch
import torch.nn as nn
from keras import Input
from torchsummary import  summary
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、函数功能区 -----------------------------------
inp1 = Input(shape=(32,32,3))
inp2 = Input(shape=(64,64,3))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),# 卷积(32,32,3------32,32,16)
            torch.nn.ReLU(),                                                                # 激活
            torch.nn.MaxPool2d(2,stride=2),                                                 # 池化(32,32,16------16,16,16)

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),# 卷积(16,16,16------16,16,32)
            torch.nn.ReLU(),                                                                     # 激活
            torch.nn.MaxPool2d(2, stride=2),                                                     # 池化(16,16,32-----8,8,32)

        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(2048, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(8192, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.ReLU())
    def forward1(self,x):
        output = self.conv1(x)
        output1 = self.fc1(output)
        return output1
    def forward1(self,x):
        output2=self.fc2(x)
        return output2

model = Net()
print(model)
#----------------------------------------------------------------------------------------------------------------------#
# 自定义loss
class customloss(nn.Module):
    def __init__(self):
        super(customloss,self).__init__()
    def forward(self,y_true,y_pred):
        loss = torch.mean(torch.abs(y_true-y_pred))
        return loss

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = customloss()
