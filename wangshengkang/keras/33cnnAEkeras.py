# -*- coding: utf-8 -*-
# @Time: 2020/6/3 15:27
# @Author: wangshengkang

# -----------------------------------代码布局--------------------------------------------
# 1引入keras，numpy，matplotlib，IPython等包
# 2导入数据，数据预处理
# 3建立模型
# 4训练模型，预测结果
# 5结果以及损失函数可视化
# -----------------------------------代码布局--------------------------------------------
# ------------------------------------1引入包-----------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, add
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理------------------------------------------

path = 'mnist.npz'
f = np.load(path)
print(f.files)

X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)  # (60000, 28, 28)
print(X_test.shape)  # (10000, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32') / 255.  # 归一化
X_test = X_test.astype('float32') / 255.

print('X_train shape', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------
x = Input(shape=(28, 28, 1))  # 1*28*28
conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)  # 16*28*28
pool1 = MaxPooling2D((2, 2), padding='same')(conv1)  # 16*14*14
conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)  # 8*14*14
pool2 = MaxPooling2D((2, 2), padding='same')(conv2)  # 8*7*7
conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)  # 8*7*7
h = MaxPooling2D((2, 2), padding='same')(conv3)  # 8*4*4

conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)  # 8*4*4
up1 = UpSampling2D((2, 2))(conv4)  # 8*8*8
conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)  # 8*8*8
up2 = UpSampling2D((2, 2))(conv5)  # 8*16*16
conv6 = Conv2D(16, (3, 3), activation='relu')(up2)  # 16*14*14
up3 = UpSampling2D((2, 2))(conv6)  # 16*28*28
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)  # 1*28*28

autoencoder = Model(inputs=x, outputs=r)  # 完整的模型
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------

epochs = 2
batch_size = 128

history = autoencoder.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(X_test, X_test)
                          )
decoded_imgs = autoencoder.predict(X_test)  # 打印输出层效果，查看解码效果

# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5结果可视化------------------------------------------
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
