# ----------------开发者信息--------------------------------
# 开发者：姜媛
# 开发日期：2020年6月23日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------


#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 -------------------------------


#  --------------------- 2、读取数据及与图像预处理 ---------------------
path = 'C:\\Users\\HP\\Desktop\\每周代码学习\\迁移学习\\数据集'
# 函数：数据加载
def load_data():
    paths = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    # 将文件解压并划分为数据集
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()     # 加载数据集
# 由于mist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,因为扩充一下mnist的最后一维
'''
cv2.resize：将原图放大到48*48
cv2.cvtColor(p1,p2) ：是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_GRAY2RGB 将灰度图片转化为格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
'''
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

# 转换为tensor形式
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)
#  --------------------- 2、读取数据及与图像预处理 ---------------------


#  ---------------------------------3、参数定义 --------------------------------
batch_size = 32
num_classes = 10
epochs = 5
num_predictions = 20
#  ----------------------------------- 3、参数定义---------------------------------------------


# -------------------------------4、模型构建------------------------
model = models.vgg16(pretrained=True)  # 使用VGG16的权重

# 特征层中参数都固定住，不会发生梯度的更新
for parma in model.parameters():
    parma.requires_grad = False

# pytorch输入图片的尺寸必须是CxHxW，所以使用premute方法把[60000, 1, 28, 28]变为[60000, 28, 28, 1]
x_train =x_train.permute(0, 3 , 2, 1)

# 重新定义最后的三个全连接层，也就是分类层，7*7*512是vgg16最后一个卷积层的输出大小
model.classifier = torch.nn.Sequential(torch.nn.Linear(7 * 7 * 512, 256),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(256, 10),
                                       torch.nn.Softmax()
                                       )

print(model)  # 查看模型
# -------------------------------4、模型构建------------------------


# -------------------------------5、模型训练------------------------
# 使用gpu
model = model.cuda(device='1')
x_train = x_train.cuda()
y_train = y_train.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # 优化器
loss_func = torch.nn.CrossEntropyLoss()                   # 损失函数

print("-----------训练开始-----------")

for i in range(epochs):
    # 预测结果
    pred = model(x_train)
    # 计算损失
    loss = loss_func(pred, y_train.long())
    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 梯度更新
    optimizer.step()
    print(i, loss.item())

print("-----------训练结束-----------")
torch.save(model.state_dict(), "torch_transferlearning.pkl")  # 保存模型参数
# -------------------------------5、模型训练------------------------


#  -------------------------- 6、模型测试 -------------------------------
print("-----------测试开始-----------")
model.load_state_dict(torch.load('torch_transferlearning.pkl')) # 加载训练好的模型参数
for i in range(epochs):
    # 预测结果
    pred = model(x_test)
    # 计算损失
    loss = loss_func(pred, y_test.long())
    # 打印迭代次数和损失
    print(i, loss.item())
print("-----------测试结束-----------")
# -------------------------------6、模型测试------------------------
