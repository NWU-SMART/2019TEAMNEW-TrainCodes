# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/18
#  @Author: MiJizong
#  @Content: GAN——Pytorch方法修改
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相应的包
# 2、读取手图像数据及与图像预处理
# 3、超参数设置
# 4、定义生成器
# 5、定义辨别器
# 6、查看模型
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块显卡
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手图像数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train = f['x_train']
# 测试数据
X_test = f['x_test']
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)  # 输出X_test维度   (10000, 28, 28)

img_rows, img_cols = 28, 28

# 数据预处理
#  归一化
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

print(np.min(X_train), np.max(X_train))
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    0.0 1.0
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 进行格式转变
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
#  --------------------- 2、读取手图像数据及与图像预处理 ---------------------

#  --------------------- 3、超参数设置 -------------------------------------

# shape、丢弃率设置
shp = X_train.shape[1:]  # 图片尺寸为 (1, 28, 28)
dropout_rate = 0.25

# 生成1 * 28 * 28的图片
nch = 200
#  --------------------- 3、超参数设置 --------------------------------------

#  --------------------- 4、定义生成器 --------------------------------------

# CNN生成图片

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100,nch*14*14),  # 100维的0~1之间的高斯分布 -> 200*14*14
            nn.ReLU(),

            nn.Upsample(size=(28,28)), # 200*28*28
            nn.Conv2d(nch*28*28,100,(3,3),padding=1),  # 200 * 28 * 28 --> 100 * 28 * 28
            nn.ReLU(),

            nn.Conv2d(100,50,(3,3),padding=1),  # 100 * 28 * 28 --> 50 * 28 * 28
            nn.ReLU(),

            nn.Conv2d(50,1,(1,1),padding=1),  # 50 * 28 * 28 --> 1 * 28 * 28
            nn.Sigmoid()
        )
    def forward(self,g_input):
        return self.gen(g_input)

# 生成generator模块
G = Generator()
#generator.compile(loss='binary_crossentropy', optimizer=opt)  # Adam(lr=1e-4)
#generator.summary()  # 输出模型各层的参数状况

#  --------------------- 4、定义生成器 ---------------------


#  --------------------- 5、定义判别器 ---------------------

# 辨别是否来自真实训练集
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1,256,(5,5),stride=(2,2),padding=1),  # 1*28*28 -> 256*28*28
            nn.LeakyReLU(0.2),  # 给所有负值赋予一个非零斜率0.2
            nn.Dropout(dropout_rate),

            nn.Conv2d(256,512,(5,5),stride=(2,2),padding=1),  # 256 -> 512
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Flatten(),  # 512 * 7 * 7 --> 25088   将数据“压平”,即把多维变为一维

            nn.Linear(25088,256),   # 25088 -> 256
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rate),

            nn.Linear(256,2),
            nn.Softmax()
        )
    def forward(self,d_input):
        return self.disc(d_input)

# 判别discriminator模块
D = Discriminator()
#discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)  # Adam(lr=1e-5)
#discriminator.summary()


#  --------------------- 5、定义辨别器 --------------------------
#  --------------------- 6、查看模型 ----------------------------
print(G)
print(D)
#  --------------------- 6、查看模型 ----------------------------

'''
#  --------------------- 6、构造生成对抗网络 ---------------------

criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(G.parameters(),lr=0.0003)
d_optimizer = torch.optim.Adam(D.parameters(),lr=0.0003)

# 冷冻训练层(定义make_trainable函数。在交替训练过程中，不需要训练辨别器，在训练生成器时)
def make_trainable(net, val):
    net.trainable = val  # false
    for l in net.layers:
        l.trainable = val


make_trainable(discriminator, False)

# 构造GAN
gan_input = Input(shape=[100])  # 输入数据
H = generator(gan_input)  # 生成新的图像，生成器  1 * 100 --> 1 * 28 * 28
gan_V = discriminator(H)  # 判别器  1 * 28 * 28  --> 1 * 2 (输入 [0 1]为真实图像, [1 0]为生成图像)

# 输入gan_input，生成图像，然后判别器判别输出结果为gan_V （整体GAN网络包括：生成器和判别器）
GAN = Model(gan_input, gan_V)

# GAN的Loss
GAN.compile(loss='categorical_crossentropy', optimizer=opt)  # Adam(lr=1e-4)

# GAN网络结果输出
GAN.summary()


#  --------------------- 6、构造生成对抗网络 ---------------------


#  --------------------- 7、训练 ---------------------

# 描绘损失收敛过程
def plot_loss(losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()


#  描绘生成器生成图像
def plot_gen(n_ex=16, dim=(4, 4), figsize=(10, 10)):
    noise = np.random.uniform(0, 1, size=[n_ex, 100])  # 随机输出[0,1)之间的n_ex * 100 个值
    generated_images = G.predict(noise)  # 生成器产生图片样本

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)  # 划分区域
        img = generated_images[i, 0, :, :]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域。
    plt.show()


# 抽取训练集样本 (60000个中抽取出来10000个训练)
ntrain = 10000
trainidx = random.sample(range(0, X_train.shape[0]), ntrain)
XT = X_train[trainidx, :, :, :]
print(X_train.shape)  # (60000, 1, 28, 28)
print(XT.shape)  # (10000, 1, 28, 28)

# generator （生成器）
# discriminator  （判别器）
# GAN （generator+discriminator）  （生成器+判别器）

# ------------------- 预训练辨别器  -----------------------------
# 预训练辨别器
noise_gen = np.random.uniform(0, 1, size=[XT.shape[0], 100])  # 生成XT.shape[0]个[0,1)之间的随机样本
generated_images = G.predict(noise_gen)  # 生成器产生图片样本

# 真实图像为XT，生成图像为generated_images
X = np.concatenate((XT, generated_images))  # 拼接
n = XT.shape[0]

y = np.zeros([2 * n, 2])  # 构造辨别器标签 one-hot encode
y[:n, 1] = 1  # 真实图像标签 [1 0]
y[n:, 0] = 1  # 生成图像标签 [0 1]

# 使得判别器可用
make_trainable(D, True)

# 预训练辨别器
D.fit(X, y, epochs=1, batch_size=32)
y_hat = D.predict(X)

#  计算辨别器的准确率
y_hat_idx = np.argmax(y_hat, axis=1)
y_idx = np.argmax(y, axis=1)
diff = y_idx - y_hat_idx  # 误差
n_total = y.shape[0]
n_right = (diff == 0).sum()  # 准确率

print("(%d of %d) right" % (n_right, n_total))

# 存储生成器和辨别器的训练损失
losses = {"d": [], "g": []}


# ------------------- 预训练辨别器  -----------------------------

# ---------------------- 7、训练 ---------------------

# --------------------- 8、输出训练数据 ---------------------

def train_for_n(nb_epoch=5000, plt_frq=25, BATCH_SIZE=32):
    for e in tqdm(range(nb_epoch)):

        # ---- 生成器生成样本generated_images -----
        image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE), :, :, :]  # 32个
        noise_gen = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        generated_images = generator.predict(noise_gen)  # generator 生成器
        # ---- 生成器生成样本generated_images -----

        # ------ 训练辨别器 ---------------
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2 * BATCH_SIZE, 2])
        y[0:BATCH_SIZE, 1] = 1
        y[BATCH_SIZE:, 0] = 1

        make_trainable(discriminator, True)  # 让判别器神经网络各层可用，使用判别器
        d_loss = discriminator.train_on_batch(X, y)  # discriminator 判别器训练 对一批样本进行单梯度更新
        losses["d"].append(d_loss)  # 存储辨别器损失loss
        # ------ 训练辨别器 ----------------

        # ------ 训练生成对抗网络 -----------
        noise_tr = np.random.uniform(0, 1, size=[BATCH_SIZE, 100])
        y2 = np.zeros([BATCH_SIZE, 2])
        y2[:, 1] = 1

        # 存储生成器损失loss
        make_trainable(discriminator, False)  # 关掉辨别器的训练
        g_loss = GAN.train_on_batch(noise_tr, y2)  # GAN 生成对抗网络(包括生成器和判别器)训练
        losses["g"].append(g_loss)
        # ------ 训练生成对抗网络 ------------

        # 更新损失loss图
        if e % plt_frq == plt_frq - 1:
            plot_loss(losses)
            plot_gen()  # 调用描绘生成器生成图像方法


train_for_n(nb_epoch=30, plt_frq=10, BATCH_SIZE=128)

#  --------------------- 8、输出训练数据 ---------------------
'''