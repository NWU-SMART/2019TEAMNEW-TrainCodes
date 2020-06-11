# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.11
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
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像
import keras
from keras.preprocessing.image import ImageDataGenerator

# 由于图像数据比较大，使用显卡加速训练
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
# 使用两块块显卡进行加速

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
import cv2
# 输入数据维度是(60000, 28, 28)，
# vgg16 需要三维图像,因为扩充一下mnist的最后一维
# 同时由于进行迁移学习时输入图片大小不能小于48*48所以将图片大小转换为48*48的
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test= [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
# 将数据转换类型，否则没有astype属性
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
# 将图片信息转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 归一化
x_train /= 255
x_test /= 255

# /-----------------  数据预处理 --------------------*/

# /----------------- 建立模型 --------------------*/
# 确定基底网络
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPool2D
from keras.applications import VGG16
from keras.models import Model
from keras.models import load_model
# base_model = load_model('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
# 如果下载太慢可以将该文件去github下载好了，然后存入服务器./keras/models中
base_model = VGG16(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
# 建立模型
model = Sequential()
# 将第一层数据维度变为展平
# (7,7,512)->7*7*512
model.add(Flatten(input_shape=base_model.output_shape[1:]))
# 7*7*512->256
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.5))
# 256->10（num_class）
model.add(Dense(units=10,activation='softmax'))
# 建立模型输入是VGG的输入输出加入了两层全连接的VGG16的输出
model = Model(inputs=base_model.input,outputs=model(base_model.output))
# 将VGG16的前15层冻结
for layers in model.layers[:15]:
    layers.trainable = False

opt = keras.optimizers.rmsprop(decay=1e-6,lr=1e-4)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
result = model.fit(x_train,y_train,
                   batch_size=32,
                   epochs=5,validation_data=[x_test,y_test],shuffle=True)
import matplotlib.pyplot as plt
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.savefig('acc.jpg')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.savefig('loss.jpg')
plt.show()