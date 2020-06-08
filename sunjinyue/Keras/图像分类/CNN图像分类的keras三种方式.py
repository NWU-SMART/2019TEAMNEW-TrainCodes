# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月5日
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import functools

#  -------------------------- 读取的载入和预处理 -------------------------------

def load_data():
    paths = [
        'D:\\应用软件\\研究生学习\\train-labels-idx1-ubyte.gz',
        'D:\\应用软件\\研究生学习\\train-images-idx3-ubyte.gz',
        'D:\\应用软件\\研究生学习\\t10k-labels-idx1-ubyte.gz',
        'D:\\应用软件\\研究生学习\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:           #  'rb'  以二进制格式打开一个文件用于只读。
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)   #frombuffer将data以流的形式读入转化成ndarray对象

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
batch_size = 128
num_classes = 10
epochs = 3
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')       # os.getcwd() 返回当前目录         os.path.join()地址拼接
model_name = 'keras_fashion_trained_model.h5'

# Convert class vectors to binary class matrices. 类别独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)        #(编码矩阵，编码长度)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

print(x_train.shape[1:])  # (28, 28, 1)
print(x_train.shape)  # (60000, 28, 28, 1)
print(y_train.shape)

#  -------------------------- 3、搭建模型 -------------------------------

# ----------------------------CNN_Class--------------------------------------

class CnnModel(keras.Model):        # 报错 提前指定Input_shape 不然检测不出编译？？？？？？？？？？？？？？？？？？？？？？
    def __init__(self):
        super(CnnModel, self).__init__(name='CnnModel')
        self.conv1 = Conv2D(32,(3,3),padding='same', activation='relu', input_shape=x_train.shape[1:])
        self.conv2 = Conv2D(32, (3, 3), activation="relu")
        self.maxpool1 = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.25)
        self.conv3 = Conv2D(64, (3, 3), padding="same", activation="relu")
        self.conv4 = Conv2D(64, (3, 3), activation="relu")
        self.maxpool2 = MaxPooling2D(pool_size=(2, 2))
        self.dropout2 = Dropout(0.25)
        self.flatten = Flatten()
        self.dense1 = Dense(units=512,activation="relu")
        self.dense2 = Dense(units=num_classes,activation="softmax")


    def call(self, x):  # 模型调用的代码
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        result = self.dense2(x)
        return result


model = CnnModel()

# /-----------------  Sequential --------------------*/
# model = Sequential()
# model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'))
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(MaxPooling2D((2,2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))

# /-----------------  API模式 --------------------*/
# from keras.layers import Input
# from keras import Model
# input = Input(shape=x_train.shape[1:])
# x = Conv2D(32,(3,3),padding='same',activation='relu')(input)
# x = Conv2D(32,(3,3),activation='relu')(x)
# x = MaxPooling2D((2,2))(x)
# x = Dropout(0.25)(x)
#
# x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
# x = Conv2D(64,(3,3),activation='relu')(x)
# x = MaxPooling2D((2,2))(x)
# x = Dropout(0.25)(x)
#
# x = Flatten()(x)
# x = Dense(512,activation='relu')(x)
# x = Dropout(0.5)(x)
# result = Dense(10,activation='softmax')(x)
# model = Model(inputs=input,outputs=result)

#----------------------------------------------------------------
opt = keras.optimizers.rmsprop(lr=0.0001, decay=0.000001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['acc'])

#  -------------------------- 3、搭建模型 -------------------------------

#  -------------------------- 4、训练 -------------------------------

if not data_augmentation:
    print('无图像增强。')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('有图像增强。')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差。
        zca_whitening=False,  # 对输入数据施加ZCA白化
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # 整数，数据提升时图片随机转动的角度。随机选择图片的角度，是一个0~180的度数，取值为0~180
        width_shift_range=0.1,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
        height_shift_range=0.1,  # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。
        shear_range=0.,  # 浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度
        zoom_range=0.,
        # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。用来进行随机的放大
        channel_shift_range=0.,  # 浮点数，随机通道偏移的幅度
        fill_mode='nearest',  # ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
        cval=0.,  # 浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        horizontal_flip=True,  # 布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
        vertical_flip=False,  # 进行随机竖直翻转
        rescale=None,  # 将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
        preprocessing_function=None,
        # 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
        data_format=None,
        # “channel_first”或“channel_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channel_last”
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )


#  -------------------------- 4、训练 -------------------------------

#  -------------------------- 5、保存模型 -------------------------------

model.summary()
# Save model and weights
if not os.path.isdir(save_dir):   #os.path.isdir()用于判断某一对象(需提供绝对路径)是否为目录
    os.makedirs(save_dir)         #创建路径
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#  -------------------------- 5、保存模型 -------------------------------

#  -------------------------- 6、显示运行结果 -------------------------------

import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()

#  -------------------------- 6、保存模型，显示运行结果 -------------------------------
