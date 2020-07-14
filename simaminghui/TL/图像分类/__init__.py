# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/11 001123:54
# 文件名称：__init__.py
# 开发工具：PyCharm
import gzip

import keras
import numpy as np

import cv2


path = ['D:\DataList\images\\train-images-idx3-ubyte.gz',
        'D:\DataList\images\\train-labels-idx1-ubyte.gz',
        'D:\DataList\images\\t10k-images-idx3-ubyte.gz',
        'D:\DataList\images\\t10k-labels-idx1-ubyte.gz']


def load_data():
    with gzip.open(path[1], 'rb') as y_train_path:
        y_train = np.frombuffer(y_train_path.read(), np.uint8, offset=8)
    with gzip.open(path[0], 'rb') as x_train_path:
        x_train = np.frombuffer(x_train_path.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(path[3], 'rb') as y_test_path:
        y_test = np.frombuffer(y_test_path.read(), np.uint8, offset=8)
    with gzip.open(path[2], 'rb') as x_test_path:
        x_test = np.frombuffer(x_test_path.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


# 得到数据集
(x_train, y_train), (x_test, y_test) = load_data()

# 此时标签为单个数字，将标签改为one-hot编码，如标签2转为[0 0 1 0 0 0 0 0 0 0 0 0],标签5转为[0 0 0 0 0 1 0 0 0 0] (按照下标算的)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)---->图像颜色空间转换,cv2.COLOR_GRAY2RGB--->彩色化：灰度图像转为彩色图像,即gray to rgb,灰色图像到彩色图像。
# cv2.resize(image, image2,dsize)     图像缩放：(输入原始图像，输出新图像，图像的大小)
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

# 格式转换,由list转为numpy.ndarray
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

# 归一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print(x_train.shape)

# -------------------------------------------迁移模型建模----------------------------#
# VGG16包含13个卷积层和3个全连接层，是一个已经建立好的model，可以直接拿来用，VGG19包含16个卷积层和3个全连接层（Very Deep Convolutional Networks--->VGG）
# include_top=False,include_top是否包括后面3个全连接层，False去掉后面的全连接层。weights:初始化神经网络参数，None表示随机初始化，imagenet表示在ImageNet训练之后的参数。
# 网上下载模型参数较慢，可以下载下来放在本地（keras\models）下。代码运行时先检查本地是否存在，不存在则下载。
# 下载链接，https://pan.baidu.com/s/1UUZ5LeKneF_MXyFVtlDCag，密码apfg，（别人百度云），下载默认路径为：C:\Users\Administrator\.keras\models下
base_model = keras.applications.VGG16(include_top=False, weights='imagenet',
                                      input_shape=x_train.shape[1:])  # input_shape = (48,48,3)

# 查看输出
base_model.summary()
print(base_model.output)
print(base_model.output_shape)

model = keras.Sequential()  # 初始化
# 展平,output_shape[1:]为1*1*512。当x_train.shape[1:]=(224*224*3)时，output_shape[1:]为7*7*512。此时我们输入当x_train.shape[1:]=(48,48,3)
model.add(keras.layers.Flatten(input_shape=base_model.output_shape[1:]))
# 1*1*512-->256
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))

# base_model和model合并，即用自己定义的全连接层
model = keras.Model(inputs=base_model.input, outputs=model(base_model.output))
model.summary()
# 卷积层的参数不变，值训练全连接层
sum = 0
for i in model.layers:
    if sum < 19:
        print('我不参与训练', i)
        i.trainable = False
    else:
        print('我参与训练', i)
    sum = sum + 1

# -------------------------------------------迁移模型建模end----------------------------#


# -------------------------------------------编译模型----------------------------#

opt = keras.optimizers.adam(lr=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])
# -------------------------------------------编译模型end----------------------------#


#-----------------------------------------训练--------------------------------------#
data_augmentation = False # 是否使用数据增强

if data_augmentation:
    print('使用数据增强')
else:
    hisTory = model.fit(x_train,y_train,
                        batch_size=256,
                        epochs=5,
                        validation_split=0.1,
                        shuffle=True
                        )

#-----------------------------------------训练end--------------------------------------#

model.save('keras_fashion_trained_model.h5')

import matplotlib.pyplot as plt
plt.plot(hisTory.history['loss'])
plt.plot(hisTory.history['val_loss'])
plt.show()

plt.plot(hisTory.history['acc'])
plt.plot(hisTory.history['val_acc'])
plt.show()

result = model.evaluate(x_test,y_test)

