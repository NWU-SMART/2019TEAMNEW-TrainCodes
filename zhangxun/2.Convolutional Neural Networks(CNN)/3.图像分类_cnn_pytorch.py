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
from torch import nn
from torch.nn import functional as F
from torch import optim
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable

#  -------------------------- 1、导入需要包 -------------------------------

def plot_curve(data): #绘制下降曲线
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_image(x, y, name):
    plt.imshow(x[0][0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
    plt.title("{}: {} ".format("name", y[0].item())) #设置标题
    plt.xticks([]) #x轴坐标设置为空
    plt.yticks([]) #y轴坐标设置为空
    plt.show() #将plt.imshow()处理后的图像显示出来

def one_hot(label, depth=10): #label转onehot （独热码:有多少个状态就有多少位置，每个位置是出现的概率，第一个位置一般表示0
    # 故1., 0., 0., ..., 0., 0., 0.表示是0的概率为1）
    out = torch.zeros(label.size(0), depth).to(device)
    idx = torch.cuda.LongTensor(label).view(-1, 1).to(device)
    out.scatter_(dim=1, index=idx, value=1).to(device)
    return out

def label(one_hot): #onehot转label
    out = torch.topk(one_hot, 1)[1].squeeze(1)
    #topk:将高维数组沿某一维度(该维度共N项),选出最大(最小)的K项并排序。返回排序结果和index信息
    return out

#  -------------------------- 2、读取数据与数据预处理 -------------------------------

def load_data():

    # # 本地path
    # paths = [
    #     '../../../数据集、模型、图片/2.CNN/MNIST/train-labels-idx1-ubyte.gz',
    #     '../../../数据集、模型、图片/2.CNN/MNIST/train-images-idx3-ubyte.gz',
    #     '../../../数据集、模型、图片/2.CNN/MNIST/t10k-labels-idx1-ubyte.gz',
    #     '../../../数据集、模型、图片/2.CNN/MNIST/t10k-images-idx3-ubyte.gz'
    # ]

    # google colab上的path
    paths = [
        './MNIST/train-labels-idx1-ubyte.gz',
        './MNIST/train-images-idx3-ubyte.gz',
        './MNIST/t10k-labels-idx1-ubyte.gz',
        './MNIST/t10k-images-idx3-ubyte.gz'
    ]


    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 1, 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 1, 28, 28)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data() # we get numpy-type datas

x_train = x_train.astype('float32') # astype: 转换数组的数据类型
x_test = x_test.astype('float32') # int32、float64是Numpy库自己的一套数据类型

x_train /= 255  # 归一化
x_test /= 255  # 归一化

# -------------- 设置超参 --------------

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 数据增强
num_predictions = 20

# -------------- 设置模型、图像保存路径名 --------------

# # 本地path
# save_dir = 'E:\\软件学习\\深度学习\\postgraduate study\\数据集、模型、图片\\2.CNN\\saved_models_cnn'

# google colab上的path
save_dir = './saved_models_cnn'

model_name = 'trained_model.h5'
# H5文件是层次数据格式第5代的版本（Hierarchical Data Format，HDF5），它是用于存储科学数据的一种文件格式和库文件。

if not os.path.isdir(save_dir): # 判断是否是一个目录(而不是文件)
    os.makedirs(save_dir) # 创造一个单层目录

model_path = os.path.join(save_dir, model_name) #模型路径名

# # 本地path
# fig_save_dir = 'E:\\软件学习\\深度学习\\postgraduate study\\数据集、模型、图片\\2.CNN\\saved_figures_cnn'

# google colab上的path
fig_save_dir = './saved_figures_cnn'

fig_acc_name = 'valid_acc.png'
fig_loss_name = 'valid_loss.png'

if not os.path.isdir(fig_save_dir): # 判断是否是一个目录(而不是文件)
    os.makedirs(fig_save_dir) # 创造一个单层目录

fig_acc_path = os.path.join(save_dir, fig_acc_name) #acc图路径名
fig_loss_path = os.path.join(save_dir, fig_loss_name) #loss图路径名


# -------------- convert to tensor --------------
"""
x_train = Variable(torch.from_numpy(x_train))  # 这样才能使用gpu加速
x_test = Variable(torch.from_numpy(x_test))
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
"""

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

# -------------- 数据可视化 --------------

y_train_onehot = one_hot(y_train)

plot_image(x_train, y_train_onehot, 'train_image')

y_test_onehot = one_hot(y_test)

plot_image(x_test, y_test_onehot, 'test_image')



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
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.Softmax
        )


    def forward(self, x):

        x = self.conv_unit(x) # [b, 1, 28, 28] => [b, 64, 5 ,5]
        x = x.view(x.size(0), -1)  # flatten: [b, 64, 5 ,5] => [b, 1600]
        out = self.fc_unit(x) # [b, 1600] => [b, 10]

        return out


device = torch.device('cuda')

model = CNNModel().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

criteion = nn.CrossEntropyLoss().to(device)

#  -------------------------- 3、搭建传统CNN模型 -------------------------------

#  -------------------------- 4、训练 -------------------------------

# step3. training

model.train()

train_loss = [] #更好可视化； train_loss为list类型

x = x_train
y = y_train

x, y = x.to(device), y.to(device)

for epoch in range(10): #开始训练，range括号内为对数据集迭代的次数

        out = model(x) #正向传播
        #print("out.shape:", out.shape) #torch.Size([1800, 10])
        y_onehot = one_hot(y)
        loss = criteion(out, y_onehot) #计算误差(代价函数) MeanSquaredError均方误差

        #print("loss.shape:", loss.shape) #torch.Size([]) loss is a 0-dim tensor

        optimizer.zero_grad() #清零梯度
        loss.backward() #反向传播，计算梯度
        optimizer.step() #更新参数  w' = w - lr*grad

        train_loss.append(loss.item())
        print("epoch:", epoch, "loss:", loss.item()) #输出计算过程

print("生成loss随epoch变化曲线图...")
plot_curve(train_loss) #画出代价函数随训练次数变化曲线图

# step4. testing

model.eval()

#训练集精度

x = x_train
y = y_train

x, y = x.to(device), y.to(device)

out = model(x)
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

out = model(x)

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

#  -------------------------- 5、保存和加载模型 -------------------------------


# Save model and weights


torch.save(model.state_dict(), model_path)
print('Saved trained model at %s ' % model_path)

# load local model

model.load_state_dict(torch.load(model_path))
print("Created model and loaded weights from file at %s " % model_path)

#  -------------------------- 5、保存和加载模型 -------------------------------

#  -------------------------- 6、显示运行结果 -------------------------------

"""

# 绘制训练 & 验证的准确率值
plt.plot('accuracy')
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

"""

#  -------------------------- 6、显示运行结果 -------------------------------