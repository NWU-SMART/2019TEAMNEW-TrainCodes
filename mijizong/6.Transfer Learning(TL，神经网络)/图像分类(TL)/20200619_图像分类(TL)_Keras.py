# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/19
#  @Author: MiJizong
#  @Content: 图像分类——Keras
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取数据及与图像预处理
# 3、迁移学习建模
# 4、训练
# 5、保存模型
# 6、训练过程可视化
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
import os
from keras import applications, Input
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1个GPU

'''
keras导入VGG-16下载太慢解决办法:
手动下载h5文件notop版本“vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5”
百度网盘地址
链接: https://pan.baidu.com/s/1UUZ5LeKneF_MXyFVtlDCag
提取码: apfg
文件名改成““vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5””
GitHub下载过慢，可以在网上下载后放到C:\\Users\\xxx\\.keras\\models\\  文件夹下，然后运行就可以不用下载直接训练了
'''
#  -------------------------- 1、导入需要包 -------------------------------


#  --------------------- 2、读取数据及与图像预处理 -------------------------

path = 'D:\\Office_software\\PyCharm\\datasets\\'

# 数据集加载
def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]
    # 提取训练数据标签
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 提取训练数据
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    # 提取测试数据标签
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 提取测试数据
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


# read dataset
(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 32
num_classes = 10  # 类别
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'keras_fashion_transfer_learning_trained_model.h5'

# 将类别标签转换为独热编码格式
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 由于mnist的输入数据维度是(num, 28, 28)，vgg-16 需要三维图像,因为扩充一下mnist的最后一维
# cv2.resize(i, (48, 48)) 将原图i转换为48*48
# cv2.COLOR_GRAY2RGB 灰度图转换为RGB图像
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

# 转换为array存储
x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

# 转换为float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化

#  --------------------- 2、读取数据及与图像预处理 -------------------------


#  --------------------- 3、迁移学习建模 ----------------------------------

# 使用VGG-16模型
base_model = applications.VGG16(include_top=False,  # (include_top=False 表示 不包含最后的3个全连接层)
                                weights='imagenet',  # weights：pre-training on ImageNet
                                input_shape=x_train.shape[1:])  # 第一层需要指出图像的大小、输入尺寸元组，仅当 include_top=False 时有效
print(x_train.shape[1:])
# ************** Sequential 模型 **************
# 建立CNN模型
model1 = Sequential()
print(base_model.output)
model1.add(Flatten(input_shape=base_model.output_shape[1:]))

# 7 * 7 * 512 --> 256
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.5))

# 256 --> num_classes 10
model1.add(Dense(num_classes))
model1.add(Activation('softmax'))

# add the model on top of the convolutional base
# 输入为VGG16的数据，经过VGG16的特征层，2层全连接到num_classes输出（自己加的）
model1 = Model(inputs=base_model.input, outputs=model1(base_model.output))  # VGG16模型与自己构建的模型合并

# ************** Sequential 模型 **************
# ***************** API 模型 ******************
x = Flatten(input_shape=base_model.output_shape[1:])(base_model.output)
x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes,activation='softmax')(x)
model2 = Model(inputs=base_model.input,outputs=x)
# ***************** API 模型 ******************
# ***************class继承 模型 ****************
input3 = Input(base_model.output_shape[1:])
class TL(keras.Model):  # base_model
    def __init__(self):
        super(TL,self).__init__()
        self.flatten = keras.layers.Flatten(input_shape=base_model.output_shape[1:])
        self.dense1 = keras.layers.Dense(256,activation='relu')
        self.dropout = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(num_classes)
        self.softmax = keras.layers.Softmax()
    def call(self,input3):
        x = self.flatten(input3)
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return self.softmax(x)
model3 = TL()
# ***************class继承 模型 ****************

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model1.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# 保持VGG16的前15层权值不变，即在训练过程中不训练
for layer in model1.layers[:15]:
    layer.trainable = False
#  --------------------- 3、迁移学习建模 ----------------------------------


#  --------------------- 4、训练 -----------------------------------------

# 是否使用数据增强
if not data_augmentation:
    print('Not using data augmentation.')
    history = model1.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)  # 表示是否在训练过程中随机打乱输入样本的顺序
else:
    print('Using real-time data augmentation.')
    # This will do pre-processing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,                   # 将数据集上的输入均值设为0
        samplewise_center=False,                    # 设置每个样本的均值为0
        featurewise_std_normalization=False,        # 将输入除以数据集的标准差
        samplewise_std_normalization=False,         # 将每个输入除以它的标准差
        zca_whitening=False,                        # 应用 ZCA 白化
        zca_epsilon=1e-06,                          # 用于ZCA美白的epsilon
        rotation_range=0,                           # 在0~180°范围内随机旋转图像
        width_shift_range=0.1,                      # 水平随机移动图像
        height_shift_range=0.1,                     # 垂直随机移动图像
        shear_range=0.,                             # 随机设定剪切范围
        zoom_range=0.,                              # 随机设置变焦的范围
        channel_shift_range=0.,                     # 随机设置通道移位的范围
        fill_mode='nearest',                        # 设置输入边界之外的填充点的模式
        cval=0.,                                    # 用于fill_mode的值=“ constant”
        horizontal_flip=True,                       # 随机水平翻转图像
        vertical_flip=False,                        # 随机垂直翻转图像
        rescale=None,                               #    设置缩放比例因子（在进行任何其他转换之前应用）
        preprocessing_function=None,                # 设置将应用于每个输入的函数
        data_format=None,                           # 图像数据格式，“ channels_first”或“ channels_last”
        validation_split=0.0)                       # 保留用于验证的图像比例（严格控制在0和1之间）

    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)  # 取整
    print(x_train.shape[0] / batch_size)  # 保留小数
    # 将模型拟合到由datagen.flow()生成的批次上。
    history = model1.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                               batch_size=batch_size),
                                  # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                                  # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                                  )

#  --------------------- 4、训练 -----------------------------------------


#  --------------------- 5、保存模型 -------------------------------------

model1.summary()  # 输出模型各层的参数状况
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model1.save(model_path)
print('Saved trained model at %s ' % model_path)

#  --------------------- 5、保存模型 --------------------------------------

#  --------------------- 6、训练过程可视化 --------------------------------

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_accuracy.png')
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

#  --------------------- 6、训练过程可视化 -------------------------------