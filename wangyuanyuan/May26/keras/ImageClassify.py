#--------------------------------------开发者信息---------------------------------
#开发人：王园园
#开发日期：2020.1.1 MLP-房价预测
#开发软件：pycharm
#开发项目:图像分类（keras）

#--------------------------------------代码布局-----------------------------------
#1、导入包
#2、数据导入
#3、数据预处理
#4、构建模型
#5、训练模型
#6、保存模型、显示运行结果

#----------------------------------------数据导入及数据处理----------------------------------
import gzip
from operator import le

import numpy as np
import os

from keras import Sequential, Input, Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator
from networkx.drawing.tests.test_pylab import plt
from tensorflow import keras, shape


def loadData():
    paths = [
        'D:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'D:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:                #解压paths[1]压缩包，取出训练数据的标签
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:               #解压paths[1]压缩包，取出训练数据
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2], 'rb') as lbpath:                #解压paths[2]压缩包，取出测试数据的标签
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:                #解压paths[3]压缩包，取出测试数据
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train),(x_test, y_test) = loadData()
batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'

#对标签进行独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255   #归一化
x_test /= 255    #归一化

#------------------------------------------搭建传统CNN模型-------------------------------
#------------------------------------------Sequential()--------------------------------
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#----------------------------------------------API类型-------------------------------------
input = Input(shape(0))
x = Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:])(input)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes)(x)
x = Activation('softmax')(x)
model1 = Model(inputs=input, outputs=x)

#-------------------------------------------------类继承-----------------------------------
input2 = Input(shape=())
class imageClassify(keras.Model):
    def __init__(self):
        super(imageClassify, self).__init__(name='CNN')
        self.conv1 = keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu')
        self.conv2 = keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.dropout1 = keras.layers.Dropout(0.25)
        self.conv3 = keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.dropout2 = keras.layers.Dropout(0.25)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512)
        self.activation1 = keras.layers.Activation('relu')
        self.dropout3 = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(num_classes)
        self.activation2 = keras.layers.Activation('softmax')

    def call(self, input2):
        x = self.conv1(input2)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        x = self.activation2(x)

model3 = imageClassify()


opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)     #优化器
#-------------------------------------------数据增强,训练模型-------------------------------------
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
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0]//batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10)

#-------------------------------------------保存模型并可视化------------------------------------
#绘制训练&验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

#绘制训练&验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show
