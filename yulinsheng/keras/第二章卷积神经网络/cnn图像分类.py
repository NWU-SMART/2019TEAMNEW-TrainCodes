# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.5.28
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/


# /-----------------  导入需要的包 --------------------*/
import numpy as np
import gzip
# import tensorflow.python.kearas.utils import get_file
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPool2D
# 由于图像数据比较大，使用显卡加速训练
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
# 使用四块显卡进行加速

# /-----------------  导入需要的包 --------------------*/

# /-----------------  读取数据 --------------------*/
# 写入文件路径
path_trainLabel = 'train-labels-idx1-ubyte.gz'
path_trainImage = 'train-images-idx3-ubyte.gz'
path_testLabel = 't10k-labels-idx1-ubyte.gz'
path_testImage = 't10k-images-idx3-ubyte.gz'
# 将文件解压并划分为数据集
with gzip.open(path_trainLabel,'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)#通过还原成类型ndarray。
with gzip.open(path_trainImage,'rb') as Imgpath:
    x_train = np.frombuffer(Imgpath.read(),np.uint8,offset=16).reshape(len(y_train),28,28,1)
with gzip.open(path_testLabel,'rb') as lbpath_test:
    y_test = np.frombuffer(lbpath_test.read(),np.uint8,offset=8)
with gzip.open(path_testImage, 'rb') as Imgpath_test:
    x_test = np.frombuffer(Imgpath_test.read(), np.uint8, offset=16).reshape(len(y_test),28,28,1)
# /-----------------  读取数据 --------------------*/

# /-----------------  数据预处理 --------------------*/
# 将类型信息进行one-hot编码(10类)
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)

# 将图片信息转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 归一化
x_train /= 255
x_test /= 255

# /-----------------  数据预处理 --------------------*/

# /-----------------  模型建立 --------------------*/
# /-----------------  序贯模型第一种 --------------------*/
# model = Sequential()
# model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'))
# # 卷积核3*3，same表示输入图片大小和输出一样，input_shape定义输入的维度
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(MaxPool2D((2,2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))
# /-----------------  序贯模型第二种 --------------------*/
# model = Sequential([
#     Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'),
#     Conv2D(32,(3,3),activation='relu'),
#     MaxPool2D((2,2)),
#     Dropout(0.25),
#
#     Conv2D(64,(3,3),padding='same',activation='relu'),
#     Conv2D(64,(3,3),activation='relu'),
#     MaxPool2D((2,2)),
#     Dropout(0.25),
#
#     Flatten(),
#     Dense(512),
#     Activation('relu'),
#     Dropout(0.5),
#     Dense(10),
#     Activation('softmax')
# ])


# /-----------------  API模式 --------------------*/
from keras.layers import Input
from keras import Model
input = Input(shape=x_train.shape[1:])
x = Conv2D(32,(3,3),padding='same',activation='relu')(input)
x = Conv2D(32,(3,3),activation='relu')(x)
x = MaxPool2D((2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
x = Conv2D(64,(3,3),activation='relu')(x)
x = MaxPool2D((2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
result = Dense(10,activation='softmax')(x)
model = Model(inputs=input,outputs=result)

# /----------------- 类继承的方式 --------------------*/
# import keras(需要指定一个input否则会报错误说提前编译)
# class cnn(keras.Model):
#     def __init__(self):
#         super(cnn,self).__init__(name='cnn')
#         self.conv2d1 = Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu')
#         self.conv2d2 = Conv2D(32,(3,3),activation='relu')
#         self.maxpool1 = MaxPool2D((2,2))
#
#         self.conv2d3 = Conv2D(64,(3,3),padding='same',activation='relu')
#         self.conv2d4 = Conv2D(64,(3,3),activation='relu')
#         self.maxpool2 = MaxPool2D((2,2))
#
#         self.flatten = Flatten()
#         self.dense1 = Dense(512,activation='relu')
#         self.dropout1 = Dropout(0.25)
#         self.dropout2 = Dropout(0.5)
#         self.dense2 = Dense(10,activation='softmax')
#     def call(self,x):
#         x = self.conv2d1(x)
#         x = self.conv2d2(x)
#         x = self.maxpool1(x)
#         x = self.dropout1(x)
#
#         x = self.conv2d3(x)
#         x = self.conv2d4(x)
#         x = self.maxpool2(x)
#         x = self.dropout1(x)
#
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.dropout2(x)
#         result = self.dense2(x)
#         return result
# 优化函数
optimize = keras.optimizers.rmsprop(lr=1e-4,decay=1e-6)

model.compile(loss='categorical_crossentropy',optimizer=optimize,metrics=['accuracy'])
# /-----------------  模型建立 --------------------*/

# /-----------------  模型训练 --------------------*/
# 通过判断是否使用图片增强技术去训练图片
data_augmentation = True
if not data_augmentation:
    print("没有使用图像增强技术")
    history = model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),
                    shuffle=True)#随机打乱样本顺序
else:
    print("使用了图像增强技术")
    data = ImageDataGenerator(featurewise_center = False,#将输入数据的均值设为0
                              samplewise_center = False,#将每个样本的均值设为0
                              featurewise_std_normalization = False,#逐个特征将输入数据除以标准差
                              samplewise_std_normalization = False,#将每个输入除以标准差
                              zca_whitening=False,#是否使用zca白化（降低输入的冗余性）
                              # ZCA白化主要用于去相关性，且尽量使白化后的数据接近原始输入数据
                              zca_epsilon=1e-06,#利用阈值构建低通滤波器对输入数据进行滤波
                              rotation_range=0,#随机旋转的度数
                              width_shift_range=0.1,#水平随机移动宽度的0.1
                              height_shift_range=0.1,#垂直随机移动高度的0.1
                              shear_range=0.,#不随机剪切
                              zoom_range=0.,#不随机缩放
                              channel_shift_range=0.,#通道不随机转换
                              fill_mode='nearest',#边界以外的点的填充模式aaaaaaaa|abcd|dddddddd
                              # 靠近哪个就用哪个填充，如靠近a那么就用a填充，靠近d就用d填充
                              cval=0.,#边界之外点的值
                              horizontal_flip=True,#随机水平翻转
                              vertical_flip=False,#不随机垂直翻转
                              rescale=None,#不进行缩放（若不为0和None将数据乘以所提供的值）
                              preprocessing_function=None,#应用于输入的函数
                              data_format=None,#图像数据格式
                              validation_split=0.0#用于验证图像的比例
                              )
    data.fit(x_train)
    print(x_train.shape[0] // 32)  # 取整
    print(x_train.shape[0] / 32)  # 保留小数
    history = model.fit_generator(data.flow(x_train, y_train,
                                     batch_size=32),
                        # 按batch_size大小从x,y生成增强数据
                        epochs=5,#训练批次
                        steps_per_epoch=x_train.shape[0]//32,#每个批次训练的样本
                        validation_data=(x_test, y_test),#验证集选择
                        workers=10  # 最大进程数
                       )
# /-----------------  模型训练 --------------------*/


# /-----------------  结果显示 --------------------*/

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('losss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()