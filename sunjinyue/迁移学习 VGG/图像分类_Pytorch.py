# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月23日
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
import gzip
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import cv2
import matplotlib.pyplot as plt



#  --------------------- 读取数据及与图像预处理 ---------------------

path = 'D:\\应用软件\\研究生学习\\'

def load_data():
    paths = [
        path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz',
        path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz'
    ]

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

# read dataset
(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 256
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20


X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

x_train = torch.LongTensor(x_train)
x_test = torch.LongTensor(x_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#  --------------------- 迁移学习建模 ---------------------
model = models.vgg16(pretrained=True)  # 使用VGG16的权重

# 特征层中参数都固定住，不会发生梯度的更新
for parma in model.parameters():
    parma.requires_grad = False

# pytorch输入图片的尺寸必须是CxHxW，所以使用premute方法把[60000, 28, 28, 1]变为[60000, 1, 28, 28]
x_train =x_train.permute(0,3,2,1)

# 重新定义最后的三个全连接层，也就是分类层，7*7*512是vgg16最后一个卷积层的输出大小
model.classifier = torch.nn.Sequential(torch.nn.Linear(7*7*512, 256),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(256, 10),
                                       torch.nn.Softmax())

print(model)  # 查看模型

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()


#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)

#   ---------------------- 训练模型 --------------------------
loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (x_train, y_train) in enumerate(loader):
        train_prediction = model(x_train)
        loss = loss_func(train_prediction, y_train)  # 计算损失
        loss_list.append(loss) # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        running_loss += loss.item()
    else:
        print(f"第{t}代训练损失为：{running_loss/len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
