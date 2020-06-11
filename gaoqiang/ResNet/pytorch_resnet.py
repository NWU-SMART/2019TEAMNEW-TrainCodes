# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.11
# 开发框架：pytorch
# 代码功能：ResNet18,34.50.101,152
#----------------------------------------------------------------------------------------------------------------------#
import torch.nn as nn

def conv3x3(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,padding=1,bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.covn1 = conv3x3(inplanes,planes,stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)       # inplace=True意思是是否将得到的值计算得到的值覆盖之前的值
                                                # 这样做可以节省运算内存，不用多存储其他变量
        self.covn2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride= stride

    def forward(self,x):
        residual = x

        out = self.covn1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.covn2(out)
        out = self.bn2(out)

        # 把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
#  这里可以看到resnet18和resnet34用的是基础版block，因为此时网络还不深，不太需要考虑模型的效率，而当网络加深到52，101，
#  152层时则有必要引入bottleneck结构，方便模型的存储和计算
class BottLeneck(nn.Module):
    expansion = 4  # 放大倍数
    def __init__(self,inplanes,planes,stride=1,downsample=None):
        super(BottLeneck,self).__init__()
        self.covn1 = nn.Conv2d(inplanes,planes,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.covn2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.covn3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride= stride

    def forward(self,x):
        residual = x

        out = self.covn1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.covn2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.covn3(out)
        out = self.bn3(out)

        # 把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    expansion = 4  # 放大倍数
    def __init__(self,block,layers,num_classes = 10):
        self.inplanes =64
        super(ResNet,self).__init__()
        self.covn1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False) # 因为mnist为（1，28，28）灰度图，因此输入通道数为1
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        self.avgpool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,planes,blocks,stride=1):
        # downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None
        if stride !=1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride,bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.covn1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)  # 拉平
        x = self.fc(x)

        return x


resnet18 = ResNet(BasicBlock,[2,2,2,2]) # 4层，每层两个BasicBlock堆叠
print(resnet18)
resnet34 = ResNet(BasicBlock,[3,4,6,3])# 4层，每层分别3,4,6,3个BasicBlock堆叠
print(resnet34)
resnet50 = ResNet(BottLeneck,[3,4,6,3])# 4层，每层分别3,4,6,3个BottLeneck堆叠
print(resnet50)
resnet101 = ResNet(BottLeneck,[3,4,23,3])# 4层，每层分别3,4,23,3个BottLeneck堆叠
print(resnet101)
resnet152 = ResNet(BottLeneck,[3,8,36,3])# 4层，每层分别3,8,36,3个BottLeneck堆叠
print(resnet152)






