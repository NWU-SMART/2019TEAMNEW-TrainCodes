# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/3 000310:44
# 文件名称：keras自编码
# 开发工具：PyCharm
import numpy
from keras import Input, Model
from keras.layers import Dense

path = "D:\DataList\mnist\mnist.npz"
f = numpy.load(path)
print(f.files)  # ['x_test', 'x_train', 'y_train', 'y_test']
x_train, x_test = f['x_train'], f['x_test']
f.close()

# 数据归一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 将图像从矩阵转为向量
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

# 创建模型,采用函数
input = Input(shape=(28 * 28,))
# 进行编码
hidden_1 = Dense(128, activation='relu')(input)  # 将28*28的向量进行操作得到128维的向量
h = Dense(64, activation='relu')(hidden_1)  # 将128维的向量通过全连接神经网络进行压缩得到64维的向量
# 进行解码
hidden_2 = Dense(128, activation='relu')(h)  # 64为向量通过网络得到128维向量
r = Dense(784, activation='sigmoid')(hidden_2)  # sigmoid 得到（0,1）区间的数字

model = Model(inputs=input, outputs=r)
# 编译
model.compile(optimizer='adam', loss='mse')

# 训练
history = model.fit(x_train, x_train, epochs=5, batch_size=128, validation_data=(x_test, x_test), verbose=1)

# 查看自编码效果
conv_encoder = Model(input, h)
encoded_imgs = conv_encoder.predict(x_train)

# 打印0张压缩后的图片
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(encoded_imgs[i].reshape(4, 16).T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# 查看自编码器的解码
decoded_imgs = model.predict(x_test)
plt.figure(figsize=(20, 6))
for i in range(10):
    # 原图
    ax = plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

     # 解码图
    ax = plt.subplot(3, 10, i + 10 + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))  # 784 转换为 28*28大小的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# 训练可视化

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()