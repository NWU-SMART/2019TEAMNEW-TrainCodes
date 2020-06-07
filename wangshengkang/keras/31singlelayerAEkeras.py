# -*- coding: utf-8 -*-
# @Time: 2020/5/31 16:29
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
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理------------------------------------------

f = np.load('mnist.npz')  # 导入数据
print(f.files)
X_train = f['x_train']
X_test = f['x_test']
f.close()

print(X_train.shape)
print(X_test.shape)

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
# ------------------------------------2数据处理------------------------------------------
# ------------------------------------3建立模型------------------------------------------
input_size = 784  # 输入层神经元个数
hidden_size = 64  # 隐藏层神经元个数
output_size = 784  # 输出层神经元个数

x = Input(shape=(input_size,))
h = Dense(hidden_size, activation='relu')(x)
r = Dense(output_size, activation='sigmoid')(h)

autoencoder = Model(inputs=x, outputs=r)  # 完整的模型
autoencoder.compile(optimizer='adam', loss='mse')

# 可视化模型
# 直接取得 pydot.Graph 对象并自己渲染它。 例如，ipython notebook 中的可视化实例
SVG(model_to_dot(autoencoder).create(prog='dot', format='svg'))
# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型，预测结果------------------------------------------
epochs = 5
batch_size = 125

history = autoencoder.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_test, X_test)
                          )

conv_encoder = Model(x, h)  # 编码器模型部分
encoded_imgs = conv_encoder.predict(X_test)
# ------------------------------------4训练模型，预测结果------------------------------------------
# ------------------------------------5结果可视化------------------------------------------
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)
    plt.imshow(encoded_imgs[i].reshape(4, 16).T)  # 打印编码器输出的结果，查看图片的压缩效果
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

decoded_imgs = autoencoder.predict(X_test)  # 打印输出层效果，查看解码效果

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

# 损失函数可视化
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# ------------------------------------5结果可视化------------------------------------------
