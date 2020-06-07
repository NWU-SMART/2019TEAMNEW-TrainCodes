# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/5
#  @Author: MiJizong
#  @Content: 图像分类CNN——Keras三种方法实现
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： --------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： --------------------------------------

#  -------------------------- 1、导入需要包 --------------------------------
from keras import Input, Model
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块显卡
#  -------------------------- 1、导入需要包 ---------------------------------


#  -------------------------- 2、读取数据与数据预处理 ------------------------

# 数据集和代码放一起即可
def load_data():
    paths = [
        'D:\\Office_software\\PyCharm\\datasets\\train-labels-idx1-ubyte.gz',
        'D:\\Office_software\\PyCharm\\datasets\\train-images-idx3-ubyte.gz',
        'D:\\Office_software\\PyCharm\\datasets\\t10k-labels-idx1-ubyte.gz',
        'D:\\Office_software\\PyCharm\\datasets\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:                       # 解压paths[0]中的数据，并取出训练标签
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)  # 参数offset为读取的起始位置，默认为0

    with gzip.open(paths[1], 'rb') as imgpath:                      # 解压paths[1]中的数据，并取出训练数据
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)  # 28 * 28

    with gzip.open(paths[2], 'rb') as lbpath:                       # 解压paths[2]中的数据，并取出测试标签
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:                      # 解压paths[3]中的数据，并取出测试数据
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 32
num_classes = 10    # 预测的类别个数
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')   # 模型保存
model_name = 'keras_fashion_trained_model.h5'

# Convert class vectors to binary class matrices. 类别独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')  # 转换数据类型
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

#  -------------------------- 2、读取数据与数据预处理 -------------------------

#  -------------------------- 3.1、Sequential模型 ------------------------------
model1 = Sequential()

model1.add(Conv2D(32, (3, 3), padding='same',  # 32 feature map，(3,3)是卷积核数量和大小
                 input_shape=x_train.shape[1:]))  # 第一层需要指出图像的大小
model1.add(Activation('relu'))
model1.add(Conv2D(32, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Conv2D(64, (3, 3), padding='same'))
model1.add(Activation('relu'))
model1.add(Conv2D(64, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))

model1.add(Flatten())
model1.add(Dense(512))  # 隐层  512个神经元
model1.add(Activation('relu'))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes))  # 输出层 10
model1.add(Activation('softmax'))
#  -------------------------- 3.1、Sequential模型 ------------------------------

#  -------------------------- 3.2、API模型 -------------------------------------
input2 = Input(shape=x_train.shape[1:])
x = Conv2D(32,(3,3),padding='same')(input2)
x = Activation('relu')(x)
x = Conv2D(32,(3,3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64,(3,3),padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64,(3,3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes)(x)
x = Activation('softmax')(x)
model2 = Model(inputs=input2,outputs=x)
#  -------------------------- 3.2、API模型 -------------------------------------

#  -------------------------- 3.3、class继承模型 --------------------------------
input3 = Input(shape=())
class ImageClassify(keras.Model):
    def __init__(self):
        super(ImageClassify,self).__init__()
        self.conv1 = keras.layers.Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:], activation='relu')
        self.conv2 = keras.layers.Conv2D(32,(3,3),activation='relu')
        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=(2,2))
        self.dropout1 = keras.layers.Dropout(0.25)
        self.conv3 = keras.layers.Conv2D(64,(3,3),padding='same',activation='relu')
        self.conv4 = keras.layers.Conv2D(64,(3,3),activation='relu')
        self.maxpool2 = keras.layers.MaxPooling2D(pool_size=(2,2))
        self.dropout2 = keras.layers.Dropout(0.25)
        self.flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(512)
        self.activation1 = keras.layers.Activation('relu')
        self.dropout3 = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(num_classes)
        self.activation2 = keras.layers.Activation('softmax')

    def call(self,input3):
        x = self.conv1(input3)
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
        return x
model3 = ImageClassify()
#  -------------------------- 3.3、class继承模型 --------------------------------

#  -------------------------- 4、模型编译 ---------------------------------------
# initiate RMSprop optimizer  初始化RMSprop优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 定义学习率 衰减率

# Let's train the model using RMSprop
model1.compile(loss='categorical_crossentropy',  # 交叉熵损失
              optimizer=opt,                    # 使用RMSprop优化器
              metrics=['accuracy'])
#  -------------------------- 4、模型编译 ---------------------------------------

#  -------------------------- 5、训练 -------------------------------------------
# 判断是否使用数据增强
if not data_augmentation:
    print('Not using data augmentation.')
    history = model1.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # 预处理和数据增强:
    datagen = ImageDataGenerator(   # 图像数据生成器
        featurewise_center=False,  # 将数据集上的输入均值初始化为0
        samplewise_center=False,   # 将每个样本的均值初始化为0
        featurewise_std_normalization=False,  # 将数据除以标准差
        samplewise_std_normalization=False,   # 将每个输入除以标准差
        zca_whitening=False,     # 是否使用 ZCA 白化
        zca_epsilon=1e-06,       # 定义参数epsilon 用于 ZCA 白化
        rotation_range=0,        # 随机旋转0-180°范围内的图像
        width_shift_range=0.1,   # 水平随机移动图像
        height_shift_range=0.1,  # 竖直随机移动图像
        shear_range=0.,          # 设定随机剪切范围，0不随机
        zoom_range=0.,           # 设置随机变焦的范围，0不随机
        channel_shift_range=0.,  # 设置随机通道移位的范围，0
        fill_mode='nearest',     # 设置输入边界之外的填充点的模式，靠近填充
        cval=0.,                 # value used for fill_mode = "constant"
        horizontal_flip=True,    # 随机水平翻转图像
        vertical_flip=False,     # 不随机竖直翻转图像
        rescale=None,            # 不缩放（在进行任何其他转换之前应用）
        preprocessing_function=None,  # 设置将应用于每个输入的功能
        data_format=None,        # 图像数据格式，“ channels_first”或“ channels_last”
        validation_split=0.0)    # 保留用于验证的图像比例（严格控制在0和1之间）

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)  # 取整
    print(x_train.shape[0] / batch_size)   # 保留小数
    # 将模型拟合到由datagen.flow()生成的对应批次上。
    history = model1.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                               batch_size=batch_size),
                                  # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                                  # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                                  )

#  -------------------------- 5、训练 ----------------------------------------

#  -------------------------- 6、保存模型 ------------------------------------

model1.summary()  # 输出模型各层的参数状况
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model1.save(model_path)
print('Saved trained model at %s ' % model_path)

#  -------------------------- 6、保存模型 ------------------------------------

#  -------------------------- 7、显示运行结果 --------------------------------

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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

#  -------------------------- 7、保存模型，显示运行结果 ------------------------
