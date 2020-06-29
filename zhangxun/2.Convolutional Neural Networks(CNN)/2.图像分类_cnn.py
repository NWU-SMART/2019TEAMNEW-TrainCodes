# ----------------开发者信息----------------------------
# 开发者：张迅
# 开发日期：2020年6月25日
# 内容：2.CNN-图像分类
# 修改内容：
# 修改者：
# ----------------开发者信息----------------------------



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
import functools
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第3块显卡
#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、读取数据与数据预处理 -------------------------------

# 数据集和代码放一起即可
def load_data():
    paths = [
        '../../../数据集、模型、图片/2.CNN/MNIST/train-labels-idx1-ubyte.gz',
        '../../../数据集、模型、图片/2.CNN/MNIST/train-images-idx3-ubyte.gz',
        '../../../数据集、模型、图片/2.CNN/MNIST/t10k-labels-idx1-ubyte.gz',
        '../../../数据集、模型、图片/2.CNN/MNIST/t10k-images-idx3-ubyte.gz'
    ]

    # numpy.frombuffer(buffer, dtype=float, count=-1, offset=0)

    # Parameters:
    # buffer : buffer_like
    # An object that exposes the buffer interface.
    #
    # dtype : data-type, optional
    # Data-type of the returned array; default: float.
    #
    # count : int, optional
    # Number of items to read. -1 means all data in the buffer.
    #
    # offset : int, optional
    # Start reading the buffer from this offset (in bytes); default: 0.

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
epochs = 5
data_augmentation = True  # 数据增强
num_predictions = 20

save_dir = 'E:\\软件学习\\深度学习\\postgraduate study\\数据集、模型、图片\\2.CNN\\saved_models_cnn' #模型路径文件夹
model_name = 'keras_fashion_trained_model_test.h5' #模型文件名
# H5文件是层次数据格式第5代的版本（Hierarchical Data Format，HDF5），它是用于存储科学数据的一种文件格式和库文件。

# Convert class vectors to binary class matrices. 类别独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32') # astype: 转换数组的数据类型
x_test = x_test.astype('float32') #int32、float64是Numpy库自己的一套数据类型

x_train /= 255  # 归一化
x_test /= 255  # 归一化

# ------- 数据可视化 -------

import matplotlib.pyplot as plt
import torch

# print(x_train.shape, y_train.shape) # (60000, 28, 28, 1) (60000, 10)
# print(x_train.type, y_train.type) #numpy.ndarray

x = torch.tensor(x_train)
y = torch.tensor(y_train)
x = x.squeeze()
y = torch.topk(y, 1)[1].squeeze(1) # one-hot转label
# print(x.shape, y.shape) # torch.Size([60000, 28, 28]) torch.Size([60000])

plt.imshow(x[0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} ".format("train image", y[0].item())) #设置标题
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来

x = torch.tensor(x_test)
y = torch.tensor(y_test)
x = x.squeeze()
y = torch.topk(y, 1)[1].squeeze(1) # one-hot转label

plt.imshow(x[0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} ".format("test image", y[0].item())) #设置标题
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来

# ------- 数据可视化 -------


#  -------------------------- 2、读取数据与数据预处理 -------------------------------

#  -------------------------- 3、搭建传统CNN模型 -------------------------------

model = Sequential()

# pytorch和keras在卷积层的区别:

# pytorch:
# torch.nn.Conv2d(in_channels,
#        		  out_channels,
#       		  kernel_size,
#         		  stride=1,
#         		  padding=0,
#         		  dilation=1,
#         		  groups=1,
#         		  bias=True)

# keras:
# keras.layers.Conv2D(filters, # 整数，输出空间的维度 (out_channels)
#                     # 当使用该层作为模型第一层时，需要提供 input_shape 参数 (in_channels)
#                     kernel_size, # 一个整数，或者 2 个整数表示的元组或列表
#                     strides=(1, 1), # 一个整数，或者 2 个整数表示的元组或列表
#                     padding='valid', # "SAME" = with zero padding  "VALID" = without padding
#                     data_format=None,
#                     dilation_rate=(1, 1),
#                     activation=None, # 要使用的激活函数，如果你不指定，则不使用激活函数 (即线性激活： a(x) = x)
#                     use_bias=True, # 布尔值，该层是否使用偏置向量
#                     kernel_initializer='glorot_uniform',
#                     bias_initializer='zeros',
#                     kernel_regularizer=None,
#                     bias_regularizer=None,
#                     activity_regularizer=None,
#                     kernel_constraint=None,
#                     bias_constraint=None)


model.add(Conv2D(32, (3, 3), padding='same',  # 32，(3,3)是卷积核数量和大小
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

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#  -------------------------- 3、搭建传统CNN模型 -------------------------------

#  -------------------------- 4、训练 -------------------------------

if not data_augmentation:
    print('Not using data augmentation.')

    # fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，
    # 如果有验证集的话，也包含了验证集的这些指标变化情况

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
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
    print(x_train.shape[0]//batch_size)  # 取整 1875
    print(x_train.shape[0]/batch_size)  # 保留小数 1875.0
    # Fit the model on the batches generated by datagen.flow().
    # 利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。
    # 函数返回一个History对象

    # fit_generator(self,
    #               generator, #生成器函数, 输出应该为一个形如（inputs, targets,sample_weight）的tuple
    #               steps_per_epoch,
    #               epochs=1, #整数，数据迭代的轮数
    #               verbose=1,
    #               callbacks=None,
    #               validation_data=None, # 生成验证集的生成器 或 一个形如（inputs,targets[,sample_weights]）的tuple
    #               validation_steps=None,
    #               class_weight=None,
    #               max_q_size=10,
    #               workers=1,
    #               pickle_safe=False,
    #               initial_epoch=0)
    # datagen.flow:
    # 接收numpy数组和标签为参数, 生成经过数据提升或标准化后的batch数据, 并在一个无限循环中不断的返回batch数据

    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),  
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size, #地板除法，即除完向下取整
                        validation_data=(x_test, y_test), #验证集
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )


#  -------------------------- 4、训练 -------------------------------

#  -------------------------- 5、保存模型 -------------------------------

model.summary()
# Save model and weights
if not os.path.isdir(save_dir): # 判断是否是一个目录(而不是文件)
    os.makedirs(save_dir) # 创造一个单层目录
model_path = os.path.join(save_dir, model_name) #组合成一个路径名
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# load local model
from keras.models import load_model

model = load_model(model_path)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
print("Created model and loaded weights from file at %s " % model_path)

#  -------------------------- 5、保存模型 -------------------------------

#  -------------------------- 6、显示运行结果 -------------------------------

save_dir = 'E:\\软件学习\\深度学习\\postgraduate study\\数据集、模型、图片\\2.CNN\\saved_figures_cnn'
if not os.path.isdir(save_dir): # 判断是否是一个目录(而不是文件)
    os.makedirs(save_dir) # 创造一个单层目录
fig_acc_name = 'tradition_cnn_valid_acc.png'
fig_loss_name = 'tradition_cnn_valid_loss.png'
fig_acc_path = os.path.join(save_dir, fig_acc_name)
fig_loss_path = os.path.join(save_dir, fig_loss_name)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy']) # 由于keras库版本的更新，将acc改为accuracy
plt.plot(history.history['val_accuracy']) # 由于keras库版本的更新，将val_acc改为val_accuracy
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left') # plt.legend: 给图加上图例
#plt.savefig('tradition_cnn_valid_acc.png') #默认保存在当前工作目录下
plt.savefig(fig_acc_path)
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left') # plt.legend: 给图加上图例
#plt.savefig('tradition_cnn_valid_loss.png') # 默认保存在当前工作目录下
plt.savefig(fig_loss_path)
plt.show()

#  -------------------------- 6、显示运行结果 -------------------------------