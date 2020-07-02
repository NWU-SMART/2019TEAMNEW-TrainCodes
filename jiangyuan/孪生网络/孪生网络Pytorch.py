# ----------------开发者信息--------------------------------#
# 开发者：姜媛
# 开发日期：2020年7月1日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#


#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
#  -------------------------- 1、导入需要包 -------------------------------


# ---------------------2、函数功能区-------------------------
class IFilerNet(nn.Module):
    def __init__(self):
        super(IFilerNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, ut_channels=24, kernel_size=3, stride=1, padding=1),  # 1*28*28 --> 24*28*28
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),  # 24*28*28 --> 24*14*14

            # 24*14*14 --> 64*14*14
            torch.nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            # 64*14*14--> 96*14*14
            torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU()
        )

        self.flatteen = torch.nn.Flatten()

        self.dense = torch.nn.Sequential(
                     torch.nn.Linear(18816, 512),
                     torch.nn.ReLU()
                                        )

    def forward1(self, x):
        x = self.conv(x)
        x = self.flatteen(x)
        x = self.dense(x)
        return x

    def forward2(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


model = iFilerNet()
print(model)
# ---------------------2、函数功能区-------------------------
