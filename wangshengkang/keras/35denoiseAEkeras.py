# -*- coding: utf-8 -*-
# @Time: 2020/6/8 14:08
# @Author: wangshengkang

# -----------------------------------代码布局--------------------------------------------
# 1引入keras，numpy，matplotlib，IPython等包
# 2导入数据，数据预处理
# 3建立模型
# 4训练模型，预测结果
# 5结果以及损失函数可视化
# -----------------------------------代码布局--------------------------------------------
# ------------------------------------1引入包-----------------------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils

# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理-----------------------------------------

path = 'mnist.npz'
f = np.load(path)

X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)  # (60000, 28, 28)
print(X_test.shape)  # (10000, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------
x = Input(shape=(28, 28, 1))  # 1*28*28
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 28*28*32
pool1 = MaxPooling2D((2, 2), padding='same')(conv1)  # 14*14*32
conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)  # 14*14*32
h = MaxPooling2D((2, 2), padding='same')(conv2)  # 7*7*32

conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)  # 7*7*32
up1 = UpSampling2D((2, 2))(conv3)  # 14*14*32
conv4 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14*14*32
up2 = UpSampling2D((2, 2))(conv4)  # 28*28*32
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28*28*1

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------

epochs = 3
batch_size = 128

history = autoencoder.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(X_test_noisy, X_test)
                          )

# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5结果可视化------------------------------------------
decoded_imgs = autoencoder.predict(X_test_noisy)  # 打印输出层效果，查看解码效果

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))  # 打印测试集真实图片
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))  # 打印解码后的图片
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
# ------------------------------------5结果可视化------------------------------------------
