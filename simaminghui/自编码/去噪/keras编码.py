# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/6 000613:11
# 文件名称：keras
# 开发工具：PyCharm
import numpy
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

f = numpy.load('D:\DataList\mnist\mnist.npz')
x_train, x_test = f['x_train'], f['x_test']
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# 加入噪声数据
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * numpy.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * numpy.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = numpy.clip(x_train_noisy, 0., 1.)  # [0,1]的数保留下来，小于0的数赋值为0，大于1的数赋值为1
x_test_noisy = numpy.clip(x_test_noisy, 0., 1.)  # [0,1]的数保留下来，小于0的数赋值为0，大于1的数赋值为1

# --------------------去噪编码-----------------------------------------
x = Input(shape=(28, 28, 1))
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 28*28*28-->28*28*32
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)  # 28*28*32-->14*14*32
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)  # 14*14*32-->14*14*32
h = MaxPooling2D((2, 2), padding='same')(conv1_2)  # 14*14*32-->7*7*32

# 解码器
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)  # 7*7*32-->7*7*32
up1 = UpSampling2D((2, 2))(conv2_1)  # 7*7*32-->14*14*32
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14*14*32-->14*14*32
up2 = UpSampling2D((2, 2))(conv2_2)  # 14*14*32-->28*28*32
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28*28*32-->28*28*1

model = Model(x, r)
model.compile(optimizer='adadelta', loss='binary_crossentropy')

# 训练
history = model.fit(x_train, x_train,
                    batch_size=128,
                    epochs=3,
                    verbose=1,
                    validation_data=(x_test, x_test))

# 查看效果
decoded_imgs = model.predict(x_test_noisy)

plt.figure(figsize=(20, 6))
for i in range(10):
    # 原图
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    # 最终效果
    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
plt.show()

history = history.history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.show()
