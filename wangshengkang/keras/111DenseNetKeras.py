# -*- coding: utf-8 -*-
# @Time: 2020/6/22 11:27
# @Author: wangshengkang
# -------------------------------------------代码布局----------------------------------------------------
# 1导入相关包
# 2读取数据
# 3建立模型
# -------------------------------------------代码布局----------------------------------------------------
# -------------------------------------------1引用相关包----------------------------------------------------
import numpy as np
import keras
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
import pickle
import os

# -------------------------------------------1引用相关包----------------------------------------------------
# -------------------------------------------2读取数据-------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '5'  # 选择gpu
# python的pickle模块实现了基本的数据序列和反序列化。通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存
# 到文件中去，永久存储；通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。
# 从file中读取一个字符串，并将它重构为原来的python对象。file:类文件对象，有read()和readline()接口。
data_batch_1 = pickle.load(open('cifar-10-batches-py/data_batch_1', 'rb'), encoding='bytes')
data_batch_2 = pickle.load(open('cifar-10-batches-py/data_batch_2', 'rb'), encoding='bytes')
data_batch_3 = pickle.load(open('cifar-10-batches-py/data_batch_3', 'rb'), encoding='bytes')
data_batch_4 = pickle.load(open('cifar-10-batches-py/data_batch_4', 'rb'), encoding='bytes')
data_batch_5 = pickle.load(open('cifar-10-batches-py/data_batch_5', 'rb'), encoding='bytes')

train_X_1 = data_batch_1[b'data']
train_X_1 = train_X_1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_1 = data_batch_1[b'labels']

train_X_2 = data_batch_2[b'data']
train_X_2 = train_X_2.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_2 = data_batch_2[b'labels']

train_X_3 = data_batch_3[b'data']
train_X_3 = train_X_3.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_3 = data_batch_3[b'labels']

train_X_4 = data_batch_4[b'data']
train_X_4 = train_X_4.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_4 = data_batch_4[b'labels']

train_X_5 = data_batch_5[b'data']
train_X_5 = train_X_5.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
train_Y_5 = data_batch_5[b'labels']

train_X = np.row_stack((train_X_1, train_X_2))
train_X = np.row_stack((train_X, train_X_3))
train_X = np.row_stack((train_X, train_X_4))
train_X = np.row_stack((train_X, train_X_5))

train_Y = np.row_stack((train_Y_1, train_Y_2))
train_Y = np.row_stack((train_Y, train_Y_3))
train_Y = np.row_stack((train_Y, train_Y_4))
train_Y = np.row_stack((train_Y, train_Y_5))
train_Y = train_Y.reshape(50000, 1).transpose(0, 1).astype('int32')
train_Y = keras.utils.to_categorical(train_Y)

test_batch = pickle.load(open('cifar-10-batches-py/test_batch', 'rb'), encoding='bytes')
test_X = test_batch[b'data']
test_X = test_X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
test_Y = test_batch[b'labels']
test_Y = keras.utils.to_categorical(test_Y)

train_X /= 255
test_X /= 255


# -------------------------------------------2读取数据-------------------------------------------------------
# -------------------------------------------3建立模型-------------------------------------------------------
# 稠密层函数
def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)

    if drop_rate: x = Dropout(drop_rate)(x)
    return x


# 稠密块函数
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=3)
    return x


# 传输层函数
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if is_max != 0:
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else:
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    return x


growth_rate = 12

inpt = Input(shape=(32, 32, 3))  # 输入数据维度

x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding='same')(inpt)
x = BatchNormalization(axis=3)(x)
x = LeakyReLU(alpha=0.1)(x)

x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)  # 第一个稠密块
x = TransitionLayer(x)  # 第一个传输层

x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)  # 第二个稠密块
x = TransitionLayer(x)  # 第二个传输层

x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)  # 第三个稠密块

x = BatchNormalization(axis=3)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='softmax')(x)

model = Model(inpt, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 计算损失
for ii in range(10):
    print('Epoch:', ii + 1)
    model.fit(train_X, train_Y, batch_size=100, epochs=1, verbose=1)
    score = model.evaluate(test_X, test_Y, verbose=1)
    print('Test loss=', score[0])
    print('Test accuracy=', score[1])

save_model(model, 'DenseNet.h5')
model = load_model('DenseNet.h5')

pred_Y = model.predict(test_X)
score = model.evaluate(test_X, test_Y, verbose=0)
print('Test loss =', score[0])
print('Test accuracy =', score[1])
# -------------------------------------------3建立模型-------------------------------------------------------
