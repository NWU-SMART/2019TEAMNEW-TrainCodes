# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/4
# 文件名称：去噪自编码器.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 与卷积自编码器类似，区别在于要对图像进行加噪预处理
# 1、导入需要的包
# 2、导入数据、图像预处理
# 3、构建自编码器模型
# 4、模型训练
# 5、查看自编码器的解码效果
# 6、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,add
from keras.layers.core import Dense,Activation,Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入数据、图像预处理 -------------------------------------------
# 数据存在本地E:\\keras_datasets\\mnist.npz
path = 'E:\\keras_datasets\\mnist.npz'

# 数据放到本地路径
(X_train,_),(X_test,_) = mnist.load_data(path)
# 数据格式进行转换
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# 数据预处理
# 归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

# 输出X_train和X_test维度
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# -------输出结果-------------
# X_train shape:(60000,28,28)
# 60000 train samples
# 10000 test samples
# -------输出结果--------------

# 加噪
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0,scale=1.0,size=X_test.shape)
# 使用截取函数将范围外的数强制转化为范围内的数。
# def clip(a, a_min, a_max, out=None):
# 将数组a中的所有数限定到范围a_min和a_max中，
# 即az中所有比a_min小的数都会强制变为a_min，a中所有比a_max大的数都会强制变为a_max.
X_train_noisy = np.clip(X_train_noisy,0.,1.)
X_test_noisy= np.clip(X_test_noisy,0.,1.)
#  -------------------------- 2、导入数据、图像预处理--------------------------------------------

#  -------------------------- 3、构建自编码器模型 -------------------------------------------
x = Input(shape=(28, 28, 1))
#编码器
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x) # 1*28*28-->16*28*28
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1) # 16*28*28-->16*14*14
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1) # 16*14*14-->8*14*14
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2) # 8*14*14-->8*7*7
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2) # 8*7*7-->8*7*7
h = MaxPooling2D((2, 2), padding='same')(conv1_3) # 8*7*7-->8*4*4

# 解码器
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h) # 8*4*4-->8*4*4
up1 = UpSampling2D((2, 2))(conv2_1) # 8*4*4-->8*8*8
conv2_2 = Conv2D(8, (3, 3),activation='relu', padding='same')(up1) # 8*8*8-->8*8*8
up2 = UpSampling2D((2, 2))(conv2_2) # 8*8*8-->8*16*16
conv2_3 = Conv2D(16, (3, 3),activation='relu')(up2) # 8*16*16-->16*14*14(not same)
up3 = UpSampling2D((2, 2))(conv2_3) # 16*14*14-->16*28*28
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3) # 16*28*28-->1*28*28

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') # 损失函数用二元交叉熵

#  -------------------------- 3、构建自编码器模型 ------------------------------------------

#  -------------------------- 4、模型训练------------------------------------------
epochs = 3
batch_size = 128
history = autoencoder.fit(X_train_noisy, X_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=2,
                          validation_data=(X_test_noisy, X_test))
#  -------------------------- 4、模型训练------------------------------------------

#  -------------------------- 5、查看自编码器的解码效果 ------------------------------------------
decoded_imgs = autoencoder.predict(X_test_noisy)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3, n, i+1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 打印解码图
    ax = plt.subplot(3,n,i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#  -------------------------- 5、查看自编码器的解码效果-------------------------------------------

# ----------------------------6、训练过程可视化------------------------------------------------
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper right')
plt.savefig('DenoiseAutoEncoder_valid_loss.png')
plt.show()
# ----------------------------6、训练过程可视化-----------------------------------------------