# -*- coding: utf-8 -*-
# @Time: 2020/6/9 15:05
# @Author: wangshengkang
# -----------------------------------代码布局--------------------------------------------
# 1引入keras，numpy，matplotlib，IPython等包
# 2导入数据，数据预处理
# 3建立模型
# 4训练模型
# 5保存模型
#6画出准确率和损失函数的变化曲线
# -----------------------------------代码布局--------------------------------------------
# ------------------------------------1引入包-----------------------------------------------
import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
import os
from keras import applications
import cv2
import matplotlib.pyplot as plt

# ------------------------------------1引入包-----------------------------------------------
# ------------------------------------2数据处理-----------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

path = './'


def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0], 'rb') as lbpath:
        # frombuffer将data以流的形式读入转化成ndarray对象
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()#读取数据

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True
num_predictions = 20

#标签one-hot编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#转变颜色空间
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255#归一化
x_test /= 255

# ------------------------------------2数据处理-----------------------------------------
# ------------------------------------3建立模型------------------------------------------
#VGG16模型， include_top=False，不包含最后的3个全连接层
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
print(x_train.shape[1:])

#建立模型
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
#VGG+自己的模型
model = Model(inputs=base_model.input, outputs=model(base_model.output))
#VGG16前15层权值固定住
for layer in model.layers[:15]:
    layer.trainable = False

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
# ------------------------------------3建立模型------------------------------------------
# ------------------------------------4训练模型------------------------------------------

if not data_augmentation:
    print('Not using data augmentation.')
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
        zca_whitening=False,
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
        validation_split=0.0
    )
    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)
    print(x_train.shape[0] / batch_size)

    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10
                                  )
# ------------------------------------4训练模型------------------------------------------
# ------------------------------------5保存模型------------------------------------------
model.summary()
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'keras_fashion_transfer_learning_trained_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s' % model_path)
# ------------------------------------5保存模型------------------------------------------
# ------------------------------------6画曲线------------------------------------------
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
# ------------------------------------6画曲线------------------------------------------