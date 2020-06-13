# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras import Input,Model
import functools



# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第3块显卡
#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、读取数据与数据预处理 -------------------------------

# 数据集和代码放一起即可
def load_data():
    paths = [
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\train-labels-idx1-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\train-images-idx3-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\t10k-labels-idx1-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:
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


(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 32
num_classes = 10
epoch = 1
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'

# Convert class vectors to binary class matrices. 类别独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

#  -------------------------- 2、读取数据与数据预处理 -------------------------------

#  -------------------------- 3、搭建传统CNN模型 -------------------------------
"""
#model = Sequential()
#model.add(Conv2D(32, (3, 3), padding='same',  # 32，(3,3)是卷积核数量和大小
                 input_shape=x_train.shape[1:]))  # 第一层需要指出图像的大小
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

inputs=Input(shape=(28,28,1))
x=Conv2D(32,kernel_size=3,padding='same',activation='relu')(inputs)
x=Conv2D(32,kernel_size=3,activation='relu')(x)
x =MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(0.25)(x)
x=Conv2D(64,kernel_size=3,padding='same',activation='relu')(x)
x=Conv2D(64,kernel_size=3,activation='relu')(x)
x =MaxPooling2D(pool_size=(2,2))(x)
x=Dropout(0.25)(x)
x=Flatten()(x)
x=Dense(512,activation='relu')(x)
x=Dropout(0.5)(x)
predict=Dense(num_classes,activation='softmax')(x)
model=keras.Model(inputs=inputs, outputs=predict, name='classfier')

"""
inputs=Input(shape=(28,28,1))
class SimpleMLP(Model):
    def __init__(self ):
       super(SimpleMLP, self).__init__()


       self.conv2d1=Conv2D(32,kernel_size=3,padding='same',activation='relu')
       self.conv2d2=Conv2D(32,kernel_size=3,activation='relu')
       self.maxpooling2d1=MaxPooling2D(pool_size=(2,2))
       self.conv2d3=Conv2D(64,kernel_size=3,padding='same',activation='relu')
       self.conv2d4=Conv2D(64,kernel_size=3,activation='relu')
       self.maxpooling2d2 = MaxPooling2D(pool_size=(2,2))
       self.dense1=Flatten()
       self.dense2=Dense(512,activation='relu')
       self.dropout=Dropout(0.5)
       self.dense3=Dense(10,activation='softmax')

    def call(self,inputs):
        x=self.conv2d1(inputs)
        x= self.conv2d2(x)
        x=self.maxpooling2d1(x)

        x=self.conv2d3(x)
        x=self.conv2d4(x)
        x=self.maxpooling2d2(x)

        x= self.dense1(x)
        x= self.dense2(x)
        x=self.dropout(x)
        return self.dense3(x)
model=SimpleMLP()
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#  -------------------------- 3、搭建传统CNN模型 -------------------------------

#  -------------------------- 4、训练 -------------------------------

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epoch,
                        validation_data=(x_test, y_test),
                        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)  # 取整
    print(x_train.shape[0] / batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                               batch_size=batch_size),
                                  # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                                  # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                                  epochs=epoch,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                                  )

#  -------------------------- 4、训练 -------------------------------

#  -------------------------- 5、保存模型 -------------------------------

model.summary()
# Save model and weights
#if not os.path.isdir(save_dir):
#    os.makedirs(save_dir)
#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)

#  -------------------------- 5、保存模型 -------------------------------

#  -------------------------- 6、显示运行结果 -------------------------------

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
"""
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()"""

# 绘制训练 & 验证的损失值


#  -------------------------- 6、保存模型，显示运行结果 -------------------------------