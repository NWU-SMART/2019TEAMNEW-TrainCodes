# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月3日
# 内容:Resnet
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
from torch.nn import Conv2d,ReLU,MaxPool2d,AvgPool2d,Linear,Softmax
import torch
class basic_block1(torch.nn.Module):#维度不变，这是基础块
    def __init__(self,in_channels):
        super(basic_block1,self).__init__()
        self.conv1 = Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
            shortcut = x
            x = self.normal(x)
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            output1 = self.relu(x + shortcut)
            return output1
class basic_block2(torch.nn.Module):#w维度改变，这是残差块
    def __init__(self,in_channels,out_channels):
        super(basic_block2, self).__init__()
        self.conv1 = Conv2d(in_channels,out_channels,kernel_size=1,stride=2)
        self.relu = ReLU()
        self.conv2 = Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
        self.conv3 = Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
    def forward(self, x):
        shortcut = self.conv1(x)
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        output2 = self.relu(x+shortcut)
        return output2
class Resnet(torch.nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxp1 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resn1 = basic_block1(64)
        self.resn2 = basic_block1(64)
        self.resn3 = basic_block2(64, 128)
        self.resn4 = basic_block1(128)
        self.rest5 = basic_block2(128, 256)
        self.rest6 = basic_block1(256)
        self.rest7 = basic_block2(256, 512)
        self.rest8 = basic_block1(512)
        self.avgp1 = AvgPool2d(7)
        self.fullc = Linear(512, 1000)
        self.relu = ReLU()
        self.softmax = Softmax()
    def forward(self, x):
        in_size = x.size(0)
        x = self.maxp1(self.relu(self.conv1(x)))
        x = self.resn1(x)
        x = self.resn2(x)
        x = self.resn3(x)
        x = self.resn4(x)
        x = self.resn5(x)
        x = self.resn6(x)
        x = self.resn7(x)
        x = self.resn8(x)
        x = self.avgp1(self.relu(x))
        x = x.view(in_size, -1)
        x = self.fullc(x)
        result = self.softmax(x)
        return  result  ###使用softmax激活函数进行得分计算
model = Resnet()
print(model)
