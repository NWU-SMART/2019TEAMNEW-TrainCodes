# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月25日
# 内容:迁移学习-图像回归
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 pytorch的包
# 2、读取数据及与图像预处理
# 3、迁移学习建模
# 4、训练
# 5、模型可视化与保存模型
# 6、训练过程可视化
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import numpy as np
import torch
import cv2
import pandas as pd
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
from keras_applications.densenet import models
path = 'D:\\northwest\\小组视频\\5单层自编码器\\mnist.npz'#用的fashion minist 数据集，有十类衣服
f = np.load(path)
X_train = f['x_train']
X_test = f['x_test']
Y_train = f['y_train']
Y_test = f['y_test']
f.close()
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
x_train = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in X_train]
x_test = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in X_test]
'''
cv2.resize：将原图放大到48*48
cv2.cvtColor(p1,p2) ：是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_GRAY2RGB 将灰度图片转化为格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片

'''
X_train = np.asarray(X_train)#转为array数组
X_test= np.asarray(X_test)#转为array数组
X_train = X_train.astype("float32")/255.#归一化
X_test = X_test.astype("float32")/255.#归一化
from torch.autograd import Variable   #从torch.autograd导入variable
X_train = Variable(torch.from_numpy(X_train))  # 将x_train变为variable数据类型，将numpy转为variable
X_test = Variable(torch.from_numpy(X_test))    # x_test变为variable数据类型
'''
Variable类型的变量具有自动求导的功能data: 取出Variable里面的tensor数值
grad：是variable的反向传播梯度，也就是求导的结果
grad_fn: 是指得到这个Variable所要进行的操作，比如是通过加减乘除
'''
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
#  ---------------------- 3、伪造回归数据 ---------------------------------
'''
这里手写体数据集有10类，假设手写体每一类是一类衣服，他有十个标签，所以可以给这10类假设一个价格，
然后对他进行回归预测，就是伪造回归数据。如何伪造，利用了正态分布给每一类标上价格
正态分布的两个重要指标是均值和标准差，决定了整个正态分布的位置和形状，所以这里利用了它的性质。
提前设定好每类的价格（均值），然后设定一个标准差，通过正态分布堆积生成价格然后赋予给对应类的衣服。
'''
#首先将数据转换成Dataframe来处理。因为它比较简单，他是一个表格型的数据类型，每列值的类型可以不同，是最常用的pandas类型
Y_train_pd = pd.DataFrame(Y_train)
Y_test_pd = pd.DataFrame(Y_test)
Y_train_pd.columns = ['label']
Y_test_pd.columns = ['label']
#设置价格，下面是均值列表
'''
sort 与 sorted 区别：
sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
'''
mean_value_list = [45,57,85,99,125,27,180,152,225,33]
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)],3,size=1))[0]#np.random.normal是正态分布，均值有均值列表给出，方差是3，输出的值放在size的shape里面
    return np.round(price,2)# 返回按指定位数进行四舍五入的数值(这里保留两位)
Y_train_pd['price'] = Y_train_pd['label'].apply(setting_clothes_price)
Y_test_pd['price'] = Y_test_pd['label'].apply(setting_clothes_price)
print(Y_train_pd.head())           # 打印前五个训练标签
print('--------------------')
print(Y_test_pd.head())            # 打印前五个测试标签
#  ---------------------- 3、伪造回归数据 ---------------------------------
#  -----------------------4、数据归一化 -----------------------------------
from sklearn.preprocessing import MinMaxScaler    # MinMaxScaler：归一到 [ 0，1 ] ；MaxAbsScaler：归一到 [ -1，1 ]
min_max_scaler = MinMaxScaler()#归一化
min_max_scaler.fit(Y_train_pd)#训练集归一化
Y_train = min_max_scaler.transform(Y_train_pd)[:,1]#归一化之后的数据
Y_train = torch.Tensor(Y_train)     #Y_train转换为Tensor类型
min_max_scaler.fit(Y_test_pd)#测试集归一化
Y_test= min_max_scaler.transform(Y_test_pd)[:,1]#归一化之后的数据
Y_test = torch.Tensor(Y_test)      #Y_test转换为Tensor类型
Y_test_label = min_max_scaler.transform(Y_test_pd)[:,0]# 归一化后的标签
print(len(Y_train))
print(len(Y_test))
#  -----------------------4、数据归一化 -----------------------------------
#  -----------------------5、迁移学习建模 ---------------------------------
import torch.nn as nn
class VGGNet(nn.Module):
    def __init__(self,num_classes=1):
        super(VGGNet,self).__init__()
        net = models.vgg16(pretrained=True)
        net.classifier = nn.Sequential()#将vgg6的分类成换成自己定义的分类层
        for parm in net.parameters():
            parm.requirs_grad = False#不计算梯度
            self.feature = net  # 保留vgg16的特征层
            self.classifier = nn.Sequential(
                nn.Linear(512*7*7,256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256,num_classes),
                nn.ReLU()

            )
    def forward(self,x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)#相当于Flatten
        x = x.classfier(x)
        return
#  -----------------------5、迁移学习建模 ---------------------------------
#  -----------------------6、训练 -----------------------------------------
model=VGGNet()
print(model)
optimizer = torch.nn.Adam(model.parameters(),lr=1e-4)
loss_fn = torch.nn.MSELoss()
Epoch=5
for i in range(5):
    y_p = model(X_train)
    loss = loss_fn(y_p,Y_train)
    if(i+1)%1==0:
        print('epoch[{}/{}],loss:{:.4f}',format(i+1,5,loss.item()))#每迭代一次，打印一次损失值
optimizer.zero_grad()#更新梯度之前，首先要进行梯度清零
loss.backward()#计算梯度
optimizer.step()#更新梯度
if(i+1)%5==0:
    torch.save(model.state_dict(),"/.pytorch_transfer-learning_model.h5")
    print("save model")
#  -----------------------6、训练 -----------------------------------------





