# ------------------------------------作者信息---------------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/28 13:35
# @Author: wangshengkang

# -------------------------------------作者信息--------------------------------------------
# -------------------------------------代码布局：---------------------------------------
# 1引入gzip,numpy,keras,os等包
# 2导入数据，处理数据
# 3创建模型
# 4训练模型
# 5保存模型
# 6画图
# -------------------------------------代码布局：---------------------------------------
# ------------------------------------1引入相关包--------------------------------------
import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
import os


# ------------------------------------1引入相关包----------------------------------
# -----------------------------------2导入数据，数据处理------------------------------------------
def load_data():
    paths = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0], 'rb') as lbpath:
        # frombuffer将data以流的形式读入转化成ndarray对象
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()
# -----------------------------------2导入数据，数据处理------------------------------------------

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'

y_train = keras.utils.to_categorical(y_train, num_classes)  # 将整型的类别标签转为onehot编码
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')  # 转换数组类型
x_test = x_test.astype('float32')

x_train /= 255  # 图像的RGB数值归一化到(0,1)
x_test /= 255
print(x_train.shape[1:])  # (28, 28, 1)
# -----------------------------------2导入数据，数据处理------------------------------------------
# -----------------------------------3创建模型----------------------------------------------
model = Sequential()
# 32个卷积核，卷积核尺寸为3*3，输入尺寸为(28, 28, 1)
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))#32*28*28
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))#32*26*26
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))#32*13*13
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))#64*13*13
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))#64*13*13
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))#64*6*6
model.add(Dropout(0.25))

model.add(Flatten())#2304
model.add(Dense(512))#512
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))#10
model.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=0.000006)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# -----------------------------------3创建模型----------------------------------------------
# -----------------------------------4训练模型--------------------------------------------------
if not data_augmentation:
    print('Not using data augmentaion.')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)
    print(x_train.shape[0] / batch_size)
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10)
# -----------------------------------4训练模型--------------------------------------------------
# -----------------------------------5保存模型--------------------------------------------------
model.summary()
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)  # 保存模型
print('Saved trained model at %s' % model_path)

# -----------------------------------5保存模型--------------------------------------------------
# -----------------------------------6画图--------------------------------------------------
import matplotlib.pyplot as plt

# 画准确率的曲线
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()
# 画损失的曲线
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Modle loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
# -----------------------------------6画图--------------------------------------------------
