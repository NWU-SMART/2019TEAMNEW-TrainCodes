# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月15日
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
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


#  --------------------- 读取手写体数据及与图像预处理 ---------------------

path = 'D:\\应用软件\\研究生学习\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 数据放到本地路径

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

# 加入噪声数据

noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)   # sss限制到 0-1 之间


#  --------------------- 构建去噪自编码器模型 ---------------------

x = Input(shape=(28, 28, 1))

# 编码器
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # 28*28*1 --> 28*28*32
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)  # 28*28*32 --> 14*14*32
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)  # 14*14*32 --> 14*14*32
h = MaxPooling2D((2, 2), padding='same')(conv1_2) # 14*14*32 --> 7*7*32


# 解码器
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)  # 7*7*32 --> 7*7*32
up1 = UpSampling2D((2, 2))(conv2_1)  # 7*7*32 --> 14*14*32
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14*14*32 --> 14*14*32
up2 = UpSampling2D((2, 2))(conv2_2)  # 14*14*32 --> 28*28*32
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28*28*32 --> 28*28*1

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(autoencoder).create(prog='dot.exe', format='svg'))


#  --------------------- 训练 ---------------------

epochs = 3
batch_size = 128

history = autoencoder.fit(X_train_noisy, X_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_test_noisy, X_test))


#  --------------------- 查看解码效果 ---------------------

# decoded_imgs 为输出层的结果
decoded_imgs = autoencoder.predict(X_test_noisy)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 解码效果图
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


#  --------------------- 训练过程可视化 --------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
