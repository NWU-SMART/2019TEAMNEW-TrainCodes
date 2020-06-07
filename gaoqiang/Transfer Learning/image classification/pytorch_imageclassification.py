# ----------------------------------------------开发者信息---------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.04
# 开发框架：pytorch
# 温馨提示：服务器上跑
#------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局--------------------------------------------------------#
# 1、加载图像数据
# 2、图像数据预处理
# 3、训练模型
# 4、保存模型与模型可视化
# 5、训练过程可视化
#-------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------加载图像数据-----------------------------------------------------#
import numpy as np
import gzip            # 使用python gzip库进行文件压缩与解压缩
def load_data():
    # 训练标签 训练图像 测试标签 测试图像
    # 本地
    # paths = [
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\train-labels-idx1-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\train-images-idx3-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-labels-idx1-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    # ]
    # 服务器
    paths = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
    ]
    # 读取训练标签(解压)
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取训练图像(解压)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    # 读取测试标签(解压)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取测试图像(解压)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
# 调用函数 获取训练数据和测试数据
(x_train, y_train), (x_test, y_test) = load_data()


#-------------------------------------------图像数据预处理-----------------------------------------------------------#
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


import torch
from torch.autograd import Variable
x_train = np.asarray(x_train)  # 变为数组
x_test = np.asarray(x_test)
x_train = x_train.astype('float32')  # 变为浮点型
x_test = x_test.astype('float32')
x_train /= 255  # 数据归一化
x_test /= 255   # 数据归一化
x_train = Variable(torch.from_numpy(x_train))  # x_train变为variable数据类型
x_test = Variable(torch.from_numpy(x_test))    # x_test变为variable数据类型
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------构建transfer leearning模型-----------------------------------------------------#

import torch.nn as nn
import torchvision.models as models
class VGGNet(nn.Module):
    def __init__(self,num_classes=10):
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
            nn.Linear(256,num_classes),# 分为10类
            nn.Softmax()
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
loss_fn = torch.nn.CrossEntropyLoss()

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
# epoch [1/5], loss:2.3035
# epoch [2/5], loss:2.2419
# epoch [3/5], loss:2.1591
# epoch [4/5], loss:2.0695
# epoch [5/5], loss:1.9942
# save model











# Loss_list =[]
# Accuracy_list = []
# ## 开始训练 ##
# for epoch in range(5):
#     print('epoch {}'.format(epoch + 1))
#     # training-----------------------------
#     train_loss = 0.
#     train_acc = 0.
#
#     out = model(x_train)
#     loss = loss_fn(out, y_train)
#     train_loss += loss.data[0]
#     pred = torch.max(out, 1)[1]      # troch.max()[1]， 只返回最大值的每个索引
#     train_correct = (pred == y_train).sum()   # 预测值和真实值相同的，求正确的
#     train_acc += train_correct.data[0]
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
#         x_train)), train_acc / (len(x_train))))
#
#     # evaluation--------------------------------
#     model.eval()
#     eval_loss = 0.
#     eval_acc = 0.
#     out = model(x_test)
#     loss = loss_fn(out, y_test)
#     eval_loss += loss.data[0]
#     pred = torch.max(out, 1)[1]
#     num_correct = (pred == y_test).sum()
#     eval_acc += num_correct.data[0]
#     print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
#         x_test)), eval_acc / (len(y_test))))
#
#     Loss_list.append(eval_loss / (len(x_test)))
# Accuracy_list.append(100 * eval_acc / (len(x_test)))
#
# x1 = range(0, 100)
# x2 = range(0, 100)
# y1 = Accuracy_list
# y2 = Loss_list
# import matplotlib.pyplot as plt
# plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-')
# plt.title('Test accuracy vs. epoches')
# plt.ylabel('Test accuracy')
# plt.subplot(2, 1, 2)
# plt.plot(x2, y2, '.-')
# plt.xlabel('Test loss vs. epoches')
# plt.ylabel('Test loss')
# plt.show()
# plt.savefig("accuracy_loss.jpg")
