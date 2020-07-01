# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月22日
# 内容：TL图像分类
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#
'''
# GPU训练
# 损失函数用L1loss
'''

#  -------------------------- 导入需要包 -------------------------------
import gzip
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

plt.style.use('ggplot') # 画的更好看


#  --------------------- 读取手写体数据及与图像预处理 ---------------------
path = 'D:\\研究生\\代码\\Keras代码\\3.AutoEncoder(自编码器)\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
x_train=f['x_train']
x_test=f['x_test']
y_train = f['y_train']
y_test = f['y_test']
f.close()

batch_size = 32
epochs = 5
data_augmentation = True

# 观察下X_train和X_test维度
print(x_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(x_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 由于mist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,因为扩充一下mnist的最后一维
x_train = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_test]
# 将数据变为array数组类型,否则后面会报错
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
# /255 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.

print('X_train shape:', x_train.shape)  # (60000, 48, 48, 3)
print(x_train.shape[0], 'train samples')  # 60000 train samples
print(x_test.shape[0], 'test samples')   # 10000 test samples

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

#  --------------------- 伪造回归数据 ---------------------

# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

# 给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]  # 均值列表
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3,size=1))[0] #均值mean,标准差std,数量
    return np.round(price, 2)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

#  --------------------- 数据归一化 ---------------------
# y_train_price_pd = y_train_pd['price'].tolist()
# y_test_price_pd = y_test_pd['price'].tolist()
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max_scaler.fit(y_test_pd)  # 我感觉去掉这一行精度会提高， 原因说不太清楚
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))


#x_train = Variable(torch.from_numpy(x_train))  # 参考学长的，要不然无法使用gpu加速
#x_test = Variable(torch.from_numpy(x_test))
#y_train = torch.LongTensor(y_train)
#y_test = torch.LongTensor(y_test)

x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)
#  --------------------- 迁移学习建模 ---------------------
model = models.vgg16(pretrained=True)  # 使用VGG16的权重

# 特征层中参数都固定住，不会发生梯度的更新
for parma in model.parameters():
    parma.requires_grad = False

# pytorch输入图片的尺寸必须是CxHxW，所以使用premute方法把[60000, 28, 28, 1]变为[60000, 28, 28, 1]
x_train =x_train.permute(0, 3, 2, 1)

# 重新定义最后的三个全连接层，也就是分类层，7*7*512是vgg16最后一个卷积层的输出大小
model.classifier = torch.nn.Sequential(torch.nn.Linear(7*7*512, 256),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(0.5),
                                       torch.nn.Linear(256, 1),
                                       torch.nn.ReLU()
                                       )

print(model)  # 查看模型

'''# 以下三行可以调用GPU加速训练，也就是在模型，x_train，y_train后面加上cuda()'''
model = model.cuda()
x_train = x_train.cuda()
y_train = y_train.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# 这里我用的是L1Loss     取预测值和真实值的绝对误差的平均数
loss_func = torch.nn.L1Loss()  # 我感觉在这里比MSE好一点


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
