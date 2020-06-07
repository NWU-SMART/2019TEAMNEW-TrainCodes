#----------------------------------2020.05.28 torch------------------
#-------------------------2020.05.29Keras的 和model 继承类，以及加噪-------------

# ----------------------   代码布局： ----------------------
# 1、导入 torch, matplotlib, numpy,  和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 8、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision

#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

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
Y_train = f['y_train']
Y_test = f['y_test']
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)  # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32") / 255.
X_test = X_test.astype("float32") / 255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)



#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、构建单层自编码器模型 ---------------------
class AutoEncoder(nn.module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 28 * 28),
            nn.Sigmoid()

        )

        def forward(self, x):
            encoder = self.encoder(x)
            decoder = self.decoder(encoder)
            return encoder, decoder



# CNN编码器

"""
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
class AutoEncoder(nn.module): 
    def __init__(self):
        super(AutoEncoder,self).__init__()
         self.conv1=nn.Sequential(
            nn.Conv2D(1,16,3,1,1),
            nn.ReLU(),
            nn.MaxPooling2D(2,2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2D(16,8,3,1,1)
            nn.ReLU(),
            nn.MaxPooling2D(2,2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2D(8,8,3,1,1)
            nn.ReLU(),
            nn.MaxPooling2D(2,2,)
        self.conv4=nn.Sequential(
            nn.Conv2D(8,8,3,1,1)
            nn.ReLU(),
            nn.MaxPooling2D(2,2,)
        self.conv5=nn.Sequential(
            nn.Conv2D(8,8,3,1,1)
            nn.ReLU(),
            nn.MaxPooling2D(2,2,)
        self.conv6=nn.Sequential(
            nn.Conv2D(8,16,3,1,1)
            nn.ReLU(),
            nn.MaxPooling2D(2,2,)
        self.conv7=nn.Sequential(
            nn.Conv2D(16,1,3,1,1)
            nn.Sigmoid()
    def forward(self,input)
        x=self.conv1(input）
        x=self.conv2(x)
        encoder=self.conv3(x)
        x=self.conv4(encoder)
        x=self.conv5(x)
        x=self.conv6(x)
        decoder=self.conv7(x)
 loss=BCEWithLogitsLoss   
 类继承
 class SimpleMLP(keras.Model):
    def __init__(self, use_bn=False, use_dp=False):
       super(SimpleMLP, self).__init__(name='mlp')
       self.use_bn = use_bn
       self.use_dp = use_dp
       self.conv2d1=keras.layers.conv2D(16,3,padding='same',activation='relu')
       self.maxpooling2d1=keras.layers.MaxPooling2D(2)
       self.cinv2d2=keras.layers.conv2D(8,3,padding='same',activation='relu')
       self.maxpooling2d2=keras.layers.MaxPooling2D(2)
       self.cinv2d3=keras.layers.conv2D(8,3,padding='AVLID',activation='relu')
       self.maxpooling2d3=keras.layers.MaxPooling2D(2)
              
       
       self.conv2d4=keras.layers.conv2D(8,3,PADDING='same',activation='relu')
       self.up1 = keras.layers.UpSampling2D(2)
       self.conv2d5=keras.layers.conv2D(8,3,padding='same'activation='relu')
       self.up1=keras.layers.UpSampling2D(2)
       self.conv2d5=keras.layers.conv2D(16,3,padding='VALID'activation='sigmoid')
       self.up3=keras.layers.UpSampling2D(2)
    def call(self,input):
        x=self.conv2d1(input)
        x=self.maxpooling2d1(x)
        x= self.conv2d2(x)
        x=self.maxpooling2d2(x)
        x=self.conv2d3(x)
        encoder=self.maxpooling2d3(x)
        
        x=self.conv2d4(encoder)
        x=self.up(x)
        x=self.conv2d5(x)
        x=self.up1(x)
        x=self.conv2d6(encoder)
        decoder=self.up3(x)
        return encoder,decoder
        
        
     
model=SimpleMLP()
 
 """
epoch = 5
batch_size = 128
autodecoder = AutoEncoder()
optimizer = torch.optim.ada(autodecoder.parameters(), lr=0.001)
loss = nn.binary_crossentropy()
for i in range(epoch):
    for j in range(len(X_train)):

        encoder, decoder = autodecoder(X_train[j])#X_train_nosiy[j]
        loss = loss(decoder, Y_train)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()
        num_correct = 0
        encoder2, decoder2 = autodecoder(X_test[j])#X_test_nosiy[j]
        predict = autodecoder(decoder, X_test[j])
        num_correct += (predict == Y_test[j]).sum()
        accuracy = num_correct() / len(X_test)
        if j % 1000 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), 'test acc:%.4f', accuracy.data.numpy())
# 打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoder2[i].reshape(4, 16).T)  # 8*8 的特征，转化为 4*16的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#  --------------------- 6、查看自编码器的压缩效果 ---------------------

#  --------------------- 7、查看自编码器的解码效果 ---------------------


# decoded_imgs 为输出层的结果


n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 打印解码图
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoder2[i].reshape(28, 28))  # 784 转换为 28*28大小的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()