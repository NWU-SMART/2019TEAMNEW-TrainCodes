# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.28
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/
import torch
from torch.nn import Sequential,Conv2d,ReLU,MaxPool2d,Linear
class model(torch.nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.cnn = Sequential(

            Conv2d(3, 16, kernel_size=3, padding=1, stride=1),  # 28*28*16
            ReLU(),
            MaxPool2d(kernel_size=2),  # 14*14*16

            Conv2d(16, 32, kernel_size=3, padding=1, stride=1),  # 14*14*32
            ReLU(),
            MaxPool2d(kernel_size=2),  # 7*7*32

            Conv2d(32, 64, kernel_size=3, padding=1, stride=1),  # 4*4*32
            ReLU(),

        )
        self.dense = Sequential(
            Linear(in_features=512,out_features=64),
            ReLU(),
            Linear(64,10)
        )
    def forward(self,x):
        out = self.cnn(x)
        outpu= out.view(out.size()[0], -1)
        out = self.dense(out)
        return out
    def forward_zong(self,x1,x2):
        out1 = self.forward(x1)
        out2 = self.forward(x2)
        return out1,out2
net = model()
print(net)
