# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/3 000316:26
# 文件名称：CNN自编码
# 开发工具：PyCharm
import numpy
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D


path = "D:\DataList\mnist\mnist.npz"
f = numpy.load(path)
x_train, x_test = f['x_train'], f['x_test']
f.close()

# 将数据进行转换
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# 归一化
x_train.astype('float32') / 255.
x_test.astype('float32') / 255.

# 构建模型
input = Input(shape=(28, 28, 1))
# 编码
# 28*28*1——→28*28*16
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input)  # 16个卷积核（深度为16），卷积核大小为3*3
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)  # 池化 28*28*16——→14*14*16,深度为16
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)  # 14*14*16——→14*14*8
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)  # 14*14*8——→7*7*8
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)  # 7*7*8——→7*7*8
h = MaxPooling2D((2, 2), padding='same')(conv1_3)  # 7*7*8——→4*4*8

# 解码
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)  # 4*4*8——→4*4*8
up1 = UpSampling2D((2, 2))(conv2_1)  # 每个像素复制四份，4*4*8——→8*8*8
conv2_2 = Conv2D(8, (3, 3), padding='same', activation='relu')(up1)  # 8*8*8——→8*8*8
up2 = UpSampling2D((2, 2))(conv2_2)  # 继续每个像素复制四份，并添加其周围 8*8*8——→16*16*8
conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)  # 16*16*8——→14*14*16
up3 = UpSampling2D((2, 2))(conv2_3)  # 14*14*16——→28*28*16
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)  # 28*28*16——→28*28*1

model = Model(input, r)
# 编译
model.compile(optimizer='adadelta', loss='binary_crossentropy')
model.summary()
history = model.fit(x_train, x_train, epochs=3, batch_size=128, validation_data=(x_test, x_test))

# 查看最终效果效果
decoded_imgs = model.predict(x_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(20,6))
# 前10个对比图
for i in range(10):
    # 原图
    ax = plt.subplot(3,10,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 经过训练后
    ax = plt.subplot(3,10,i+10+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# 查看损失
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'],loc='upper left')
plt.show()