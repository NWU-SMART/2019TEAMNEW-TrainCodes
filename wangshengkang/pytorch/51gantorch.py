# -*- coding: utf-8 -*-
# @Time: 2020/6/11 21:00
# @Author: wangshengkang
# 注：此程序目前有问题
# ----------------------------------------代码布局--------------------------------------------
# 1导入包
# 2导入数据，图像预处理
# 3超参数设置
# 4构建生成器模型
# 5构建判别器模型
# 6训练
# 7输出训练数据
# ------------------------------------------1导入包---------------------------------------------
import random
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt

# ------------------------------------------1导入包---------------------------------------------
# ------------------------------------------2导入数据，数据预处理-------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 选择gpu
plt.switch_backend('agg')  # 服务器没有gui
path = 'mnist.npz'
f = np.load(path)  # 导入数据
X_train = f['x_train']  # 训练集
X_test = f['x_test']  # 测试集
f.close()
print(X_train.shape)  # 60000*28*28

img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)  # 增加通道维度
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32") / 255.  # 归一化
X_test = X_test.astype("float32") / 255.
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

print(X_train.shape)  # 60000*1*28*28

# ------------------------------------------2导入数据，数据预处理-------------------------------------------
# ------------------------------------------3超参数设置-------------------------------------------
shp = X_train.shape[1:]  # 1*28*28
print('shp', shp)
dropout_rate = 0.25

print('channels X_train shape:', X_train.shape)  # 60000*1*28*28

nch = 200


# ------------------------------------------3超参数设置-------------------------------------------
# ------------------------------------------4定义生成器-------------------------------------------
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gnet1 = nn.Sequential(
            nn.Linear(100, 39200),  # 39200
            nn.BatchNorm1d(39200),
            nn.ReLU(),
        )
        self.gnet2 = nn.Sequential(
            nn.Upsample((28, 28)),  # 200*28*28
            nn.Conv2d(200, 100, (3, 3), padding=1),  # 100*28*28
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 50, (3, 3), padding=1),  # 50*28*28
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Conv2d(50, 1, (1, 1)),  # 1*28*28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.gnet1(x)
        x = x.view(10000, 200, 14, 14)
        x = self.gnet2(x)
        return x


gmodel = generator()
opt = torch.optim.Adam(gmodel.parameters(), lr=1e-4)


# ------------------------------------------4定义生成器-------------------------------------------
# ------------------------------------------5定义判别器-------------------------------------------
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dnet = nn.Sequential(
            nn.Conv2d(1, 256, (3, 3), padding=1),  # 256*28*28
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool2d((2, 2)),  # 256*14*14
            nn.Dropout(dropout_rate),
            nn.MaxPool2d((2, 2)),  # 256*7*7
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), padding=1),  # 512*7*7
            nn.Dropout(dropout_rate),
            nn.Flatten(),  # 25088
            nn.Linear(25088, 256),  # 256
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2),  # 2
            nn.Softmax()
        )

    def forward(self, x):
        x = self.dnet(x)
        return x


dmodel = discriminator()
dmodel = dmodel.cuda()
dopt = torch.optim.Adam(dmodel.parameters(), lr=1e-5)
dloss = nn.BCELoss()


# ------------------------------------------5定义判别器-------------------------------------------
# ------------------------------------------6训练-------------------------------------------------------

# 画生成的gan的图片
def plot_gen(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    print('save picture--------------------------------------------------------')
    plt.savefig('51gankeras.png')
    plt.tight_layout()
    plt.show()


# 预训练判别器
ntrain = 10000  # 从训练集60000个样本中抽取10000个
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)  # 随机抽取
XT = X_train[trainidx, :, :, :]
XT = XT.detach().numpy()
print('X_train.shape', X_train.shape)  # (60000, 1, 28, 28)
print('XT.shape', XT.shape)  # (10000, 1, 28, 28)

noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100])  # 生成10000个随机样本
noise_gen = torch.from_numpy(noise_gen).float()
generated_images = gmodel(noise_gen)  # 生成器根据随机样本生成图片
generated_images = generated_images.detach().numpy()

X = np.concatenate((XT, generated_images))  # XT为真实图像，generated_images为生成图像
X = torch.from_numpy(X)
print('X.shape', X.shape)  # 20000*1*28*28
X = X.cuda()
n = XT.shape[0]

y = np.zeros([2 * n, 2])  # 构造判别器标签，one-hot编码
y[:n, 1] = 1  # 真实图像标签[1 0]
y[n:, 0] = 1  # 生成图像标签[0 1]
# RuntimeError: expected dtype Double but got dtype Float
y = torch.from_numpy(y).float()
print('y', y.shape)  # 20000*2
y = y.cuda()

set = TensorDataset(X, y)
loader = DataLoader(set, batch_size=128)
# 预训练判别器
epochs = 1
for epoch in range(epochs):
    train_loss = 0.0
    for i, data in enumerate(loader):
        pre = dmodel(X)
        batch_loss = dloss(pre, y)
        dopt.zero_grad()
        batch_loss.backward()
        dopt.step()
        train_loss += batch_loss
    print('epoch {}  loss {}'.format(epoch, train_loss / len(loader)))

y_hat = dmodel(X)

# 计算判别器准确率
y_hat_idx = np.argmax(y_hat, axis=1)
y_idx = np.argmax(y, axis=1)
diff = y_idx - y_hat_idx
n_total = y.shape[0]
n_right = (diff == 0).sum()
print('(%d of %d) right' % (n_right, n_total))

# 存储生成器和判别器的训练损失
losses = {'d': [], 'g': []}

# ------------------------------------------6训练-------------------------------------------------------
# ------------------------------------------7输出训练数据-----------------------------------------------
EPOCHS = 5000
for epoch in range(EPOCHS):
    BATCH_SIZE = 32
    # 生成器生成样本
    image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]
    noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
    generated_images = gmodel(noise_gen)

    # 训练判别器
    X = np.concatenate((image_batch, generated_images))
    y = np.zeros([2 * BATCH_SIZE, 2])
    y[0:BATCH_SIZE, 1] = 1
    y[BATCH_SIZE:, 0] = 1

    # 训练生成对抗网络
    noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
    y2 = np.zeros([BATCH_SIZE, 2])
    y2[:, 1] = 1

# ------------------------------------------7输出训练数据-----------------------------------------------
