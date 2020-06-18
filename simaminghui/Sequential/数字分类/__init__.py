# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/17 001722:22
# 文件名称：__init__.py
# 开发工具：PyCharm


from keras.datasets import mnist
import numpy as np

# 本地数据路径
path = 'D:\DataList\mnist\mnist.npz'

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path=path)


# 对数据进行预处理。将其变换成网络要求的形状，并将所有值缩放到[0,1]区间，之前训练图像都保存在uint8的数组中
# 其形状为（60000，28,28），取值为[0,255]。我们需要将其变换为一个float32数组，其形状为（60000,28*28），取值为[0,1]
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float') / 255
# 对标签分类编码
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 构建网络
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
# 编译
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
# 训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

print(network.evaluate(test_images, test_labels)[1])
