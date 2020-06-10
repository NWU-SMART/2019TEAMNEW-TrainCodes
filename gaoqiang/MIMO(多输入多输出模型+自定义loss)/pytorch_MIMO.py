# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.09
# 开发框架：Pytorch
# 代码功能：MIMO:多输入多输出模型，自定义loss
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局------------------------------------------------------------#
# 1、建立模型
# 2、自定义loss
#----------------------------------------------------------------------------------------------------------------------#
import torch
import torch.nn as nn

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),# 卷积(32,32,3------32,32,16)
            torch.nn.ReLU(),                                                                # 激活
            torch.nn.MaxPool2d(2,stride=2),                                                 # 池化(32,32,16------16,16,16)

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),# 卷积(16,16,16------16,16,32)
            torch.nn.ReLU(),                                                                     # 激活
            torch.nn.MaxPool2d(2, stride=2),                                                     # 池化(16,16,32-----8,8,32)

        )

        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(2048, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.ReLU()
        )

    def forward(self,x):
        output = self.conv1(x)
        output = output.view(output.size(0), -1)  # 拉平，作用相当于Flatten
        output1 = self.fc1(output)
        return output1


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(4, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.ReLU()
        )

    def forward(self,x):
        output2 = self.fc2(x)
        return output2

model = ([Net1,Net2])
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





#----------------------------------------------------------------------------------------------------------------------#






