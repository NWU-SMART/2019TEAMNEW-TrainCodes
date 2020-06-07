#--------------------------------------------------------开发者信息------------------------------------------------------
#开发者：王园园
#开发日期：2020.6.04
#开发软件：pycharm
#开发项目：孪生网络（pytorch）

#----------------------------------------------------------导包---------------------------------------------------------
from numpy import concatenate
from torch import nn

#-------------------------------------------------------------函数功能区-------------------------------------------------
class FeatureNetwork(nn.Module):
    def __init__(self):
        super(FeatureNetwork, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 24, kernel_size=(3, 3), stride=1, padding='same'),   #第一层卷积
                                   nn.ReLU(True),
                                   nn.MaxPool2d(3, 3))
        self.conv2 = nn.Sequential(nn.Conv2d(24, 64, kernel_size=(3, 3), stride=1, padding='same'),   #第二层卷积
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 96, kernel_size=(3, 3), stride=1, padding='valid'),  #第三层卷积
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(96, 96, kernel_size=(3, 3), stride=1, padding='valid'),   #第四层卷积
                                   nn.ReLU(True))
        self.dense = nn.Sequential(nn.Linear(28*28*1, 512),                                            #全连接
                                   nn.ReLU(True))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.dense(out)
        return out

class ClassiFilerNet(nn.Module):
    # 生成度量网络和决策网络，其实matchnet是两个网络结构，一个是特征提取层（孪生），一个度量层+匹配层（统称为决策层）
    # 构造两个孪生网络:都是特征提取
    input1 = FeatureNetwork()
    input2 = FeatureNetwork()

    # 对于第二个网络各层更改名字
    for layer in input2.layers:
        layer.name = layer.name + str('_2')
    # 两个网络层向量拼接:向量进行融合，使用的是默认的sum，既简单的相加
    merge_layers = concatenate([input1.output, input2.output])

    def __init__(self):
        super(ClassiFilerNet, self).__init__()
        self.dense1 = nn.Sequential(nn.Linear(512, 1024),
                                    nn.ReLU(True),
                                    nn.Linear(1024, 256),
                                    nn.ReLU(True),
                                    nn.Linear(256, 2),
                                    nn.Softmax(True))
    def forward(self, x):
        out1 = self.dense1(x)
        return out1

#--------------------------------------------------------主调区----------------------------------------------------------
matchnet = ClassiFilerNet()
matchnet.summary()

