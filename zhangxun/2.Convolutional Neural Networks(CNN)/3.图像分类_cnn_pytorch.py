# ----------------开发者信息----------------------------
# 开发者：张迅
# 开发日期：2020年6月28日
# 内容：pytorch实现cnn图像分类
# 修改内容：
# 修改者：
# ----------------开发者信息----------------------------



#  -------------------------- 1、导入需要包 -------------------------------

import gzip
import numpy as np
import torch
from   torch import nn
from   torch.nn import functional as F
from   torch import optim
import os
import matplotlib.pyplot as plt

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、读取数据与数据预处理 -------------------------------

# 数据集和代码放一起即可
def load_data():
    paths = [
        '../../../数据集、模型、图片/2.CNN/MNIST/train-labels-idx1-ubyte.gz',
        '../../../数据集、模型、图片/2.CNN/MNIST/train-images-idx3-ubyte.gz',
        '../../../数据集、模型、图片/2.CNN/MNIST/t10k-labels-idx1-ubyte.gz',
        '../../../数据集、模型、图片/2.CNN/MNIST/t10k-images-idx3-ubyte.gz'
    ]

    # numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)

    # Parameters:
    # buffer : buffer_like
    # An object that exposes the buffer interface.
    #
    # dtype : data-type, optional
    # Data-type of the returned array; default: float.
    #
    # count : int, optional
    # Number of items to read. -1 means all data in the buffer.
    #
    # offset : int, optional
    # Start reading the buffer from this offset (in bytes); default: 0.

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

(x_train, y_train), (x_test, y_test) = load_data() # we get numpy-type datas

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 数据增强
num_predictions = 20

save_dir = 'E:\\软件学习\\深度学习\\postgraduate study\\数据集、模型、图片\\2.CNN\\saved_models_cnn' #模型路径文件夹
model_name = 'keras_fashion_trained_model_test.h5' #模型文件名
# H5文件是层次数据格式第5代的版本（Hierarchical Data Format，HDF5），它是用于存储科学数据的一种文件格式和库文件。

"""
# Convert class vectors to binary class matrices. 
# to_categorical: 将整型的类别标签转为onehot编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
"""

x_train = x_train.astype('float32') # astype: 转换数组的数据类型
x_test = x_test.astype('float32') #int32、float64是Numpy库自己的一套数据类型

x_train /= 255  # 归一化
x_test /= 255  # 归一化

# ------- 数据可视化 -------


# print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
# print(x_train.type, y_train.type) #numpy.ndarray

x = torch.tensor(x_train)
y = torch.tensor(y_train)
x = x.squeeze()
y = torch.topk(y, 1)[1].squeeze(1) # one-hot转label
# print(x.shape, y.shape) # torch.Size([60000, 28, 28]) torch.Size([60000])

plt.imshow(x[0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} ".format("train image", y[0].item())) #设置标题
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来

x = torch.tensor(x_test)
y = torch.tensor(y_test)
x = x.squeeze()
y = torch.topk(y, 1)[1].squeeze(1) # one-hot转label

plt.imshow(x[0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} ".format("test image", y[0].item())) #设置标题
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来

# ------- 数据可视化 -------


#  -------------------------- 2、读取数据与数据预处理 -------------------------------

#  -------------------------- 3、搭建传统CNN模型 -------------------------------

class CNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Dropout(0.25),
        )

        # flatten

        self.fc_unit = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.Softmax
        )

        # # use Cross Entropy Loss
        # self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):  # 类内定义函数，形参self必不可少

        x = self.conv_unit(x)
        x = x.view(x.size(0), -1)  # 将x打平成二维，进入全连接层
        # logits = F.softmax(self.fc_unit(x), dim=1)
        logits = self.fc_unit(x)

        # # [b, 10]
        # pred = F.softmax(logits, dim=1) #softmax:归一化指数函数
        # loss = self.criteon(logits, y)

        return logits

model = CNNModel()

# initiate RMSprop optimizer

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

criteion = nn.CrossEntropyLoss()

#  -------------------------- 3、搭建传统CNN模型 -------------------------------

#  -------------------------- 4、训练 -------------------------------

# step3. training

net.train()

train_loss = [] #更好可视化； train_loss为list类型

x = x_train
y = y_train


print("生成训练图片...")
plot_image(x, y, 'train image', 1799)

x, y = x.to(device), y.to(device)

for epoch in range(epoch_num): #开始训练，range括号内为对数据集迭代的次数

        out = net(x) #正向传播
        #print("out.shape:", out.shape) #torch.Size([1800, 10])
        y_onehot = one_hot(y)
        loss = criteion(out, y_onehot) #计算误差(代价函数) MeanSquaredError均方误差
        #loss = criteion(out, y)
        #print("loss.shape:", loss.shape) #torch.Size([]) loss is a 0-dim tensor

        optimizer.zero_grad() #清零梯度
        loss.backward() #反向传播，计算梯度
        optimizer.step() #更新参数  w' = w - lr*grad

        train_loss.append(loss.item())
        print("epoch:", epoch, "loss:", loss.item()) #输出计算过程

#train_loss = train_loss.cpu()
print("生成loss随epoch变化曲线图...")
plot_curve(train_loss) #画出代价函数随训练次数变化曲线图

# step4. testing

net.eval()

#训练集精度

x = x_train
y = y_train

x, y = x.to(device), y.to(device)

out = net(x)
pred = out.argmax(dim=1)

total_correct = 0
total_num = 1800

total_correct += pred.eq(y).sum().float().item()

acc = total_correct / total_num
print('train acc:', acc) #训练精度

#测试集精度

x = x_test
y = y_test

x, y = x.to(device), y.to(device)

out = net(x)

# out: [b, 10] => pred: [b]
pred = out.argmax(dim=1) # argmax:返回最大数的索引

total_correct = 0
total_num = 2062-1800

total_correct += pred.eq(y).sum().float().item()

acc = total_correct / total_num
print('test acc:', acc) #测试精度

x, y = x.cpu(), y.cpu()
print("生成测试图片...")
plot_image(x, pred, 'test image', 261)


#  -------------------------- 4、训练 -------------------------------

#  -------------------------- 5、保存模型 -------------------------------


# Save model and weights

if not os.path.isdir(save_dir): # 判断是否是一个目录(而不是文件)
    os.makedirs(save_dir) # 创造一个单层目录
model_path = os.path.join(save_dir, model_name) #组合成一个路径名

torch.save(model.state_dict(), model_path)
print('Saved trained model at %s ' % model_path)

# load local model

model.load_state_dict(torch.load(model_path))
print("Created model and loaded weights from file at %s " % model_path)

#  -------------------------- 5、保存模型 -------------------------------

#  -------------------------- 6、显示运行结果 -------------------------------

save_dir = 'E:\\软件学习\\深度学习\\postgraduate study\\数据集、模型、图片\\2.CNN\\saved_figures_cnn'
if not os.path.isdir(save_dir): # 判断是否是一个目录(而不是文件)
    os.makedirs(save_dir) # 创造一个单层目录
fig_acc_name = 'tradition_cnn_valid_acc.png'
fig_loss_name = 'tradition_cnn_valid_loss.png'
fig_acc_path = os.path.join(save_dir, fig_acc_name)
fig_loss_path = os.path.join(save_dir, fig_loss_name)

# 绘制训练 & 验证的准确率值
plt.plot(['accuracy'])
plt.plot(['val_accuracy'])

plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left') # plt.legend: 给图加上图例
#plt.savefig('tradition_cnn_valid_acc.png') #默认保存在当前工作目录下
plt.savefig(fig_acc_path)
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(['loss'])
plt.plot(['val_loss'])

plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left') # plt.legend: 给图加上图例
#plt.savefig('tradition_cnn_valid_loss.png') # 默认保存在当前工作目录下
plt.savefig(fig_loss_path)
plt.show()

#  -------------------------- 6、显示运行结果 -------------------------------