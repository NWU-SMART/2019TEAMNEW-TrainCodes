# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/13 001322:21
# 文件名称：test
# 开发工具：PyCharm
import keras

import numpy as np
from keras import Model, Input

x = np.asarray([1, 2, 3, 4, 10])
input = keras.layers.Input(shape=(1,))
output = keras.layers.Embedding(11, 10)(input)
# output = keras.layers.Flatten()(ouput)
model = keras.Model(input, output)
model.summary()
model.predict(x)
print(model.predict(x)[0])
print(model.predict(x)[0])
print(model.predict(x)[0].shape)

print(x.shape)

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x.shape)
label = keras.layers.Input(shape=(3,))
model = keras.Model(label, label)
print(model.predict(x))

print('-------------------------------------')
x = np.array(2)
print(x.shape)

print('-------------------------------------')
x = np.array([1, 2, 3])
y = np.array([4, 5, 7])
print(x)
print(y)
print(np.multiply(x, y))

print('-------------------------------------')
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([[7, 8, 9], [10, 11, 12]])
G_input = Input(shape=(3,))
g_input = Input(shape=(3,))
L = keras.layers.multiply([G_input, g_input])
model = Model([G_input, g_input], L)
model.summary()
print(model.predict([x, y]))

# -----------------------------查看效果----------------------------#
import matplotlib.pyplot as plt

x = np.random.uniform(0, 1, size=[10, 100])  # 10*100的矩阵
y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
generator_model = keras.models.load_model('generate_model_4.h5')
y_img = generator_model.predict([x, y])
# 画0-9数字的图像
plt.figure(figsize=(20, 6))
for i in range(10):
    # 生成器生成图像
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(y_img[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
# plt.savefig('generator.png')
plt.show()
