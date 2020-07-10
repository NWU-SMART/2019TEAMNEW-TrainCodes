# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/10 001010:16
# 文件名称：test
# 开发工具：PyCharm
import time

import keras
import numpy
from keras import Input, Model
from keras.layers import Reshape, Dense

x = numpy.array([[1, 2, 3, 4], [2, 3, 4, 5]])

model = keras.Sequential()
model.add(Dense(4, input_shape=(4,)))
model.add(Dense(1, activation='relu'))
print(model.summary())

path = "D:\DataList\mnist\mnist.npz"
f = numpy.load(path)
x_train = f['x_train']
image_batch = x_train[numpy.random.randint(0, 60000, size=128), :, :]

print(image_batch)

x = numpy.zeros((5, 1))
print(x.shape)
x = numpy.zeros((5))
print(x)

start = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
print(start)
end = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())

