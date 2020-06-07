# ---------------------------任梅--------------------------------
# ------------------------2020.06.03------------------------------
# -------------------------pytorch--------------------
# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手图像数据及与图像预处理
# 3、超参数设置
# 4、定义生成器
# 5、定义辨别器
# 6、训练
#--------------------------------导入需要包———————————————————
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn  as nn
import torch
from torch.optim import adam,opt,dopt

#--------------------------------导入需要包———————————————————

#  --------------------- 2、读取手图像数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'D:\\keras_datasets\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
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
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

#  --------------------- 2、读取手图像数据及与图像预处理 ---------------------


#  --------------------- 3、超参数设置 ---------------------

# 输入、隐藏和输出层神经元个数 (1个隐藏层)
num_epoch=200
d_steps=1
g_steps=1

shp = X_train.shape[1:]  # 图片尺寸为 (1, 28, 28)
dropout_rate = 0.25

# Optim优化器
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-5)

#  --------------------- 3、超参数设置 ---------------------

#  --------------------- 4、定义生成器 ---------------------

K.set_image_dim_ordering('th')  # 用theano的图片输入顺序

# 生成1 * 28 * 28的图片
nch = 200

# CNN生成图片
# 通过100维的
generator=nn.Sequential(
    nn.Linear(100, nch*14*14),
    nn.BatchNormalization(),
    nn.ReLU(),
    nn.Reshape([nch, 14, 14]),
    nn.UpSampling2D(2,2),
    nn.Conv2d(nch,100, 3, 1,1),
    nn.BatchNormalization(),
    nn.ReLU(),
    nn.Conv2d(50, 3, 1, 1),
    nn.BatchNormalization(),
    nn.ReLU(),
    nn.Conv2d(1,  1,),
    nn.Sigmoid(),
)
loss=nn.CrossEntropyLoss()
optimizer_g=opt(model.parameters(), lr=0.0001,weight_decay=0.0001)
#  --------------------- 4、定义生成器 ---------------------
#


#  --------------------- 5、定义辨别器 ---------------------
discriminator=nn.Sequential(
    nn.Conv2d(shp, 256,5, 1, 2),
    nn.eakyReLU(0.2),
    nn.Dropout(dropout_rate),
    nn.Linear(25088,256),
    nn.eakyReLU(0.2),
    nn.Dropout(dropout_rate),
    nn.Linear(256,2),
    nn.Sigmoid()
)
loss=nn.CrossEntropyLoss()
optimizer_d=dopt(model.parameters(), lr=0.0001,weight_decay=0.0001)
#  --------------------- 5、定义辨别器 ---------------------



#  --------------------- 6、训练 ---------------------
for epoch in range(num_epochs):
    for d_index in range(d_steps):
        D.zero_grad()
# 在真实数据集上训练判别器
        d_real_data = Variable(real_data(100))  # 得到(1, d_input_size)
        d_real_predict = discriminator(preprocess(d_real_data))
        d_real_loss = loss(d_real_predict, Variable(torch.ones(1)))
        d_real_loss.backward()

# 在噪声上训练判别器
        d_generate_data = Variable(fake_data(100, shp))  # (100,1)
        d_fake_data = generator(d_generate_data).detach()  # 将G隔离
        d_fake_predict = D(preprocess(d_fake_data.t()))  # 转置
        d_fake_loss = loss(d_fake_predict, Variable(torch.zeros(1)))
        d_fake_loss.backward()
        optimizer_d.step()

    for _ in range(g_steps):
         generator.zero_grad()
        # 一些噪声数据，数据的格式是（100,1）
         g_input = Variable(fake_data(100, shp))
         # 生成器去处理这些噪声数据（拟合真实的数据分布）
         g_fake_data = generator(100)
        # 用判别器去判断生成器的处理（生成）效果
         dg_fake_predict = D(preprocess(g_fake_data.t()))
         g_loss = loss(dg_fake_predict, Variable(torch.ones(1)))

         g_loss.backward()
         optimizer_g.step()
    if epoch % print_interval == 0:
        print("%s: D: %s/%s G: %s (Real: %s, Fake: %s) " % (epoch,
                                                           extract(d_real_loss)[0],
                                                           extract(d_fake_loss)[0],
                                                            extract(g_loss)[0],
                                                            normalize(extract(d_real_data)),
                                                            normalize(extract(d_fake_data))))

#  --------------------- 6、训练 ---------------------