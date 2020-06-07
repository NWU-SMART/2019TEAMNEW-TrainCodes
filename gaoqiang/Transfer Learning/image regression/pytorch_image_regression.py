# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.05
# 开发框架：pytorch
# 温馨提示：服务器上跑
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局------------------------------------------------------------#
# 1、加载手写体数据集  图像数据预处理
# 2、伪造回归数据
# 3、建立模型
# 4、保存模型与模型可视化
# 5、训练过程可视化
#----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------加载手写体数据集  图像数据预处理---------------------------------------------#
import numpy as np
# 载入数据本地：
# path = 'F:\\Keras代码学习\\keras\\keras_datasets\\mnist.npz'
# 载入数据服务器：
path = 'mnist.npz'
f = np.load(path)
print(f.files) # 查看文件内容 ['x_test', 'x_train', 'y_train', 'y_test']
# 定义训练数据 60000个
x_train = f['x_train']
# 定义训练标签
y_train = f['y_train']
# 定义测试数据 10000个
x_test = f['x_test']
# 定义测试标签
y_test = f['y_test']
f.close()
# 打印训练数据的维度
print(x_train.shape)  # (60000, 28, 28)
# 打印测试数据的维度
print(x_test.shape)   # (10000, 28, 28)

# 数据预处理
import cv2
'''
cv2.resize：将原图放大到48*48
cv2.cvtColor(p1,p2) ：是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_GRAY2RGB 将灰度图片转化为格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
'''
x_train = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_test]
# 将数据变为array数组类型
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
# 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.
import torch
from torch.autograd import Variable
x_train = Variable(torch.from_numpy(x_train))  # x_train变为variable数据类型
x_test = Variable(torch.from_numpy(x_test))    # x_test变为variable数据类型

#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------伪造回归数据----------------------------------------------------------#
# 转换成DataFrame格式方便数据处理（DataFrame是一个表格型的数据类型，每列值类型可以不同，是最常用的pandas对象。）
import pandas as pd
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

# 把0-9数字当做十类衣服，为其设置价格
mean_value_list = [45,57,85,99,125,27,180,152,225,33] # 均值列表
'''
这是的np是numpy包的缩写，np.random.normal()的意思是一个正态分布，normal这里是正态的意思。
numpy.random.normal(loc=0,scale=1e-2,size=shape) ，意义如下： 
参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
参数size(int 或者整数元组)：输出的值赋在shape里，默认为None。
'''
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)],3,size=1))[0] # 均值mean,标准差std,数量shape
    return np.round(price,2) # 返回按指定位数进行四舍五入的数值(这里保留两位)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)
print(y_train_pd.head())           # 打印前五个训练标签
print('--------------------')
print(y_test_pd.head())            # 打印前五个测试标签
#    label   price
# 0      5   23.24
# 1      0   40.96
# 2      4  123.72
# 3      1   58.55
# 4      9   32.76

#----------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------MinMaxScaler数据归一化-------------------------------------------------------#
# MinMaxScaler：归一到 [ 0，1 ] ；MaxAbsScaler：归一到 [ -1，1 ]
from sklearn.preprocessing import MinMaxScaler

# 训练标签归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:,1]
y_train = torch.Tensor(y_train) # y_train变为Tensor数据类型

# 测试集标签归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:,1]
y_test = torch.Tensor(y_test)
y_test_label = min_max_scaler.transform(y_test_pd)[:,0]# 归一化后的标签

print(len(y_train))  # 60000
print(len(y_test))   # 10000
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------建立迁移学习模型--------------------------------------------------------#
import torch.nn as nn
import torchvision.models as models
class VGGNet(nn.Module):
    def __init__(self,num_classes=1):
        super(VGGNet,self).__init__()
        net = models.vgg16(pretrained=True) #从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential() # 将分类层置空，下面加进我们的分类层
        for parma in net.parameters():
            parma.requires_grad = False  # 不计算梯度，不会进行梯度更新
        self.feature = net  # 保留vgg16的特征层
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,256), # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes),# 1
            nn.ReLU()
        )

    def forward(self,x):
        # 解决RuntimeError问题: Given groups=1, weight of size 32 1 3 3, expected input[60000, 28, 28, 1] to have 1 channels, but got 28 channels instead
        x = x.permute(0,3,1,2) # 例如  input[60000, 28, 28, 1]---->[6000,1,28,28] 交换位置
        x = self.feature(x)
        x = x.view(x.size(0),-1)  # 拉平，作用相当于Flatten
        x = self.classifier(x)

        return x

model = VGGNet()
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

Epoch = 5
## 开始训练 ##
for t in range(5):

    x = model(x_train)          # 向前传播
    loss = loss_fn(x, y_train)  # 计算损失

    if (t + 1) % 1 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(t + 1, 5, loss.item()))  # 每训练1个epoch，打印一次损失函数的值

    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度

    if (t + 1) % 5 == 0:
        torch.save(model.state_dict(), "./pytorch_imageclassification_transferlearning_model.h5")  # 每5个epoch保存一次模型
        print("save model")
#----------------------------------------------------------------------------------------------------------------------#
# 实验结果：
# epoch [1/5], loss:0.1886
# epoch [2/5], loss:2.5170
# epoch [3/5], loss:0.2111
# epoch [4/5], loss:0.2265
# epoch [5/5], loss:0.2325
# save model