# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/24
#  @Author: MiJizong
#  @Content: 图像回归——Pytorch
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相关的包
# 2、读取手写体数据及与图像预处理
# 3、伪造回归数据
# 4、数据归一化
# 5、迁移学习建模
# 6、模型训练
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import pandas as pd
import torch
import os
import cv2
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1个GPU
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
# 以上两句命令如果不添加汇报下列错误：
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
# That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do
# is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static
# linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you
# can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute,
# but that may cause crashes or silently produce incorrect results. For more information, please see
# http://www.intel.com/software/products/support/.
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
x_train = f['x_train']
y_train = f['y_train']
# 测试数据
x_test = f['x_test']
y_test = f['y_test']
f.close()
# 数据放到本地路径test

# 观察下x_train和x_test维度
print(x_train.shape)  # 输出x_train维度  (60000, 28, 28)
print(x_test.shape)  # 输出x_test维度   (10000, 28, 28)

# 由于mnist的输入数据维度是(num, 28, 28)，vgg-16 需要三维图像,因为扩充一下mnist的最后一维
# cv2.resize(i, (48, 48)) 将原图i转换为48*48
# cv2.COLOR_GRAY2RGB 灰度图转换为RGB图像
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

# 转换为array存储
x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

# 数据预处理
#  归一化
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    x_train shape: (60000, 48, 48, 3)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# # # 数据准备
# # # np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# # # len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# # # X_train 60000*784, X_test10000*784
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

batch_size = 32
epochs = 5
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、伪造回归数据 ---------------------

# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

# 给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]  # 均值列表


def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]  # 均值mean,标准差std,数量
    return np.round(price, 2)  # 四舍五入保留两位小数


y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

#  --------------------- 3、伪造回归数据 ---------------------

#  --------------------- 4、数据归一化 -----------------------

# y_train_price_pd = y_train_pd['price'].tolist()
# y_test_price_pd = y_test_pd['price'].tolist()
# 训练集归一化
min_max_scaler = MinMaxScaler()  # 将属性缩放到一个指定的最大和最小值（通常是0-1）之间
min_max_scaler.fit(y_train_pd)
y_train_label = min_max_scaler.transform(y_train_pd)[:, 1]  # 数据标准化

# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))

x_train = torch.FloatTensor(x_train)  # 其中加载要封装的张量   封装Tensor
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train_label)
y_test = torch.FloatTensor(y_test_label)

x_train = x_train.permute(0, 3, 2, 1)  # 维度顺序转换为1*28*28 解决输入维度不匹配问题
x_test = x_test.permute(0, 3, 2, 1)
#  --------------------- 4、数据归一化 ---------------------


#  --------------------- 5、迁移学习建模 ---------------------

'''
由于直接下载预训练VGG-16模型过慢，所以提前下载下训练好的vgg16-397923af.pth模型，
需要训练新数据时直接调用本地模型即可。
'''
# 加载预训练好的模型
import torchvision.models as models

# 使用VGG-16模型
model = models.vgg16(pretrained=False)  # 由于是加载的已训练好的模型，此处可以设置为False
pre = torch.load(r'F:\installment\vgg16-397923af.pth')  # 提取本地模型
model.load_state_dict(pre)  # 加载模型的原始状态以及参数

# # 使用VGG16模型
# base_model = applications.VGG16(include_top=False,   # include_top=False 表示 不包含最后的3个全连接层
#                                 weights='imagenet',  # weights：pre-training on ImageNet
#                                 input_shape=x_train.shape[1:])  # 第一层需要指出图像的大小

# # path to the model weights files.
# top_model_weights_path = 'bottleneck_fc_model.h5'
print(x_train.shape[1:])

# 建立CNN模型
model.classifier = torch.nn.Sequential(
    nn.Linear(7 * 7 * 512, 256),  # 7*7*512 → 256
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),  # 256 → 10
    nn.Softmax(1)
)

print(model)

#  --------------------- 5、迁移学习建模 ---------------------


#  --------------------- 6、模型训练 -------------------------

# 使用cuda调用GPU加速训练
model = model.cuda()
x_train = x_train.cuda()
y_train = y_train.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()

# 使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=24, shuffle=True, num_workers=0)
# shuffle将输入数据的顺序打乱，是为了使数据更有独立性
# num_workers工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据。

loss_list = []  # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (x_train, y_train) in enumerate(loader):
        train_prediction = model(x_train)
        loss = loss_func(train_prediction, y_train)     # 计算损失
        loss_list.append(loss)                          # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()                           # 梯度清零
        loss.backward()                                 # 反向传播
        optimizer.step()                                # 参数更新
        running_loss += loss.item()                     # 损失叠加
    else:
        print(f"第{t+1}轮训练损失为：{running_loss / len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()

#  --------------------- 6、模型训练 -------------------------