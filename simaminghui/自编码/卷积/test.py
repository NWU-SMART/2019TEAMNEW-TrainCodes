# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/3 000317:10
# 文件名称：test
# 开发工具：PyCharm
import numpy

# 一个矩阵
from keras import Input, Model
from keras.layers import UpSampling2D

x = numpy.array([[1,2],[3,4]])

x = x.reshape(1,2,2,1)
print(x)
input = Input(shape=(2,2,1))
out = UpSampling2D((2,2))(input)
model = Model(inputs=input,outputs=out)
y =model.predict(x)
print(y.reshape(4,4))
