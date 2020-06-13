# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.11
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/


# /-----------------  导入需要的包 --------------------*/
import numpy as np
import gzip
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像

# 由于图像数据比较大，使用显卡加速训练
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
# 使用两块块显卡进行加速

# /-----------------  导入需要的包 --------------------*/

# /-----------------  读取数据 --------------------*/
# 写入文件路径
path_trainLabel = 'train-labels-idx1-ubyte.gz'
path_trainImage = 'train-images-idx3-ubyte.gz'
path_testLabel = 't10k-labels-idx1-ubyte.gz'
path_testImage = 't10k-images-idx3-ubyte.gz'
# 将文件解压并划分为数据集
with gzip.open(path_trainLabel,'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)#通过还原成类型ndarray。
with gzip.open(path_trainImage,'rb') as Imgpath:
    x_train = np.frombuffer(Imgpath.read(),np.uint8,offset=16).reshape(len(y_train),28,28,1)
with gzip.open(path_testLabel,'rb') as lbpath_test:
    y_test = np.frombuffer(lbpath_test.read(),np.uint8,offset=8)
with gzip.open(path_testImage, 'rb') as Imgpath_test:
    x_test = np.frombuffer(Imgpath_test.read(), np.uint8, offset=16).reshape(len(y_test),28,28,1)
# /-----------------  读取数据 --------------------*/

# /-----------------  数据预处理 --------------------*/
import cv2
# 输入数据维度是(60000, 28, 28)，
# vgg16 需要三维图像,因为扩充一下mnist的最后一维
# 同时由于进行迁移学习时输入图片大小不能小于48*48所以将图片大小转换为48*48的
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test= [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
# 将数据转换类型，否则没有astype属性
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
# 将图片信息转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 归一化
x_train /= 255
x_test /= 255
import torch
# x_train = torch.FloatTensor(x_train)
# x_test = torch.FloatTensor(x_test)
from torch.autograd import Variable
x_train = Variable(torch.from_numpy(x_train))# 变为variable数据类型
y_train = Variable(torch.from_numpy(y_train))# 变为variable数据类型
x_test = Variable(torch.from_numpy(x_test))# 变为variable数据类型
y_test = Variable(torch.from_numpy(y_test))# 变为variable数据类型
x_train = x_train.permute(0, 3, 2, 1)
x_train.cuda()
y_train.cuda()
# x_test.cuda()

# /-----------------  数据预处理 --------------------*/

# /-----------------  模型建立 --------------------*/
from torchvision import models
import torch
# 存放位置C:\Users\yls43/.cache\torch\checkpoints
base_model = models.vgg16(pretrained=True,)
print(base_model)
# 将层数冻结
for parma in base_model.parameters():
    parma.requires_grad = False
# # 将分类的全连接层重写
base_model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                        torch.nn.Linear(4096,256),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(0.5),
                                       torch.nn.Linear(4096, 2),
                                        torch.nn.Softmax())
# model = base_model.cuda()
# 定义损失函数
# model = base_model()
base_model.cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimize = torch.optim.SGD(base_model.parameters(),lr=1e-4)
epoch = 1
# 内存占用太大，没法运行
for i in range(epoch):
    x_train = x_train.cuda()

    predict = base_model(x_train)
    loss = loss_fn(predict,y_train)
    optimize.zero_grad()
    loss.backward()
    optimize.step()
    print(i,loss.item())