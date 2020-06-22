# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/22
#  @Author: MiJizong
#  @Content: 图像分类——Pytorch(加载本地训练好的VGG-16模型)
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相应的包
# 2、读取数据及与图像预处理
# 3、迁移学习建模
# 4、训练模型
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import torch
import os
import cv2
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


#  --------------------- 2、读取数据及与图像预处理 ---------------------

path = 'D:\\Office_software\\PyCharm\\datasets\\'


# 数据集加载
def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]
    # 提取训练数据标签
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 提取训练数据
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    # 提取测试数据标签
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 提取测试数据
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


# read dataset
(x_train, y_train), (x_test, y_test) = load_data()

print(type(y_train))

# 由于mnist的输入数据维度是(num, 28, 28)，vgg-16 需要三维图像,因为扩充一下mnist的最后一维
# cv2.resize(i, (48, 48)) 将原图i转换为48*48
# cv2.COLOR_GRAY2RGB 灰度图转换为RGB图像
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

# 转换为array存储
x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

# 转换为float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

# 变为variable数据类型，类似于keras里的tensor，但比tensor有更多的属性
x_train = Variable(torch.from_numpy(x_train))
x_test = Variable(torch.from_numpy(x_test))
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

x_train = x_train.permute(0, 3, 2, 1)  # 维度顺序转换为1*28*28 解决输入维度不匹配问题
x_test = x_test.permute(0, 3, 2, 1)
#  --------------------- 2、读取数据及与图像预处理 ---------------------


#  --------------------- 3、迁移学习建模 ---------------------
'''
由于直接下载预训练VGG-16模型过慢，所以提前下载下训练好的vgg16-397923af.pth模型，
需要训练新数据时直接调用本地模型即可。
'''
# 加载预训练好的模型
import torchvision.models as models

# 使用VGG-16模型
model = models.vgg16(pretrained=False)  # 由于是加载的已训练好的模型，此处可以设置为False
pre = torch.load(r'F:\installment\vgg16-397923af.pth')  # 加载本地模型
model.load_state_dict(pre)

# 固定vgg16的模型参数，使之不再改变
for parma in model.parameters():
    parma.requires_grad = False

# 定义分类层
model.classifier = torch.nn.Sequential(
    nn.Linear(7 * 7 * 512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
    nn.Softmax()
)
print(model)
#  --------------------- 3、迁移学习建模 ---------------------


#  -----------------------4、训练模型-------------------------
'''# 以下三行可以调用GPU加速训练，也就是在模型，x_train，y_train后面加上cuda()'''
model = model.cuda()
x_train = x_train.cuda()
y_train = y_train.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()

# 使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)
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

#  -----------------------4、训练模型-------------------------
