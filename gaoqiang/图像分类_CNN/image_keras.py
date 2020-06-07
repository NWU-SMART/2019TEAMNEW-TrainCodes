# ----------------开发者信息--------------------------------#
# 开发者：高强
# 开发日期：2020年5月27日
# 开发框架：keras
#-----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、加载图像数据
# 3、图像数据预处理
# 4、训练模型
# 5、保存模型与模型可视化
# 6、训练过程可视化
#--------------------------------------------------------------#
# 任务介绍：
'''
图像分类：基于fashion MNIST数据的图像分类去做实验。在2017年8月份，德国研究机构ZalandoResearch在GitHub上推出了一个全新的
数据集，其中训练集包含60000个样例，测试集包含10000个样例，分为10类，每一类的样本训练样本数量和测试样本数量相同。样本都来
自日常穿着的衣裤鞋包，每个都是28×28的灰度图像，其中总共有10类标签，每张图像都有各自的标签。
'''
#  -------------------------- 导入需要包 -------------------------------
import gzip   # 使用python gzip库进行文件压缩与解压缩
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


# --------------------------------加载数据图像---------------------------------#
def load_data():
    # 训练标签 训练图像 测试标签 测试图像
    paths = [
        'F:\\Keras代码学习\\keras\\keras_datasets\\train-labels-idx1-ubyte.gz',
        'F:\\Keras代码学习\\keras\\keras_datasets\\train-images-idx3-ubyte.gz',
        'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-labels-idx1-ubyte.gz',
        'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]
    # 读取训练标签(解压)
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取训练图像(解压)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    # 读取测试标签(解压)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取测试图像(解压)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
# 调用函数 获取训练数据和测试数据
(x_train, y_train), (x_test, y_test) = load_data()

batch_size = 32            # 设置批大小为32
epochs = 1                 # 为了节省等待时间，先设置成1个epoch
data_augmentation = True   # 使用图像增强
num_predictions = 20

import os
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn') # 保存的模型路径
model_name = 'keras_fashion_trained_model.h5'            # 模型名字

#-------------------------------------------图像数据预处理---------------------------------------------#
#  将整型的类别标签转为onehot编码
'''
One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数
的索引之外，它都是零值，它被标记为1。one-hot编码要求每个类别之间相互独立，如果之间存在某种连续型的关系，或许使用
distributed respresentation（分布式）更加合适。
'''
num_classes = 10           # 分为10类
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 数据归一化
x_test /= 255   # 数据归一化



#  --------------------------------------- 构建CNN模型 -----------------------------------------------------#
####方法一：序贯模型####
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding='same',  # 输出32，(3,3)是卷积核大小
#                  input_shape=x_train.shape[1:]))  # 第一层需要指出图像的大小
# model.add(Activation('relu'))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, (3, 3), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())# 拉平
# model.add(Dense(512))# 隐层是512
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes))# 输出层是10类
# model.add(Activation('softmax'))


####方法二：函数API式#####
from keras import Model
from keras.layers import Input
inputs = Input(shape=(x_train.shape[1:]))
x = Conv2D(32, (3, 3), padding='same')(inputs)
x = Activation('relu')(x)
x = Conv2D(32,(3,3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64,(3,3),padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64,(3,3),padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes,activation='softmax')(x) # num_classes = 10

model = Model(inputs=inputs, outputs=outputs)

####方法三：class #####
# import keras
# from keras.layers import Input
# inputs = Input(shape=(x_train.shape[1:]))
# class mymodel(keras.Model):
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.Conv1 = keras.layers.Conv2D(32,(3,3),padding= 'same')
#         self.relu1 = keras.layers.Activation('relu')
#         self.Conv2 = keras.layers.Conv2D(32, (3, 3), padding='same')
#         self.relu2 = keras.layers.Activation('relu')
#         self.maxpooling1 = keras.layers.MaxPooling2D(pool_size= (2,2))
#         self.dropout1 = keras.layers.Dropout(0.25)
#
#         self.Conv3 = keras.layers.Conv2D(64, (3, 3), padding='same')
#         self.relu3 = keras.layers.Activation('relu')
#         self.Conv4 = keras.layers.Conv2D(64, (3, 3), padding='same')
#         self.relu4 = keras.layers.Activation('relu')
#         self.maxpooling2 = keras.layers.MaxPooling2D(pool_size=(2, 2))
#         self.dropout2 = keras.layers.Dropout(0.25)
#
#         self.Flatten = keras.layers.Flatten()
#         self.dense1 = keras.layers.Dense(512,activation='relu')
#         self.dropout3 = keras.layers.Dropout(0.5)
#         self.dense2 = keras.layers.Dense(10, activation='softmax')
#
#
#     def call(self,inputs):
#         x = self.Conv1(inputs)
#         x = self.relu1(x)
#         x = self.Conv2(x)
#         x = self.relu2(x)
#         x = self.maxpooling1(x)
#         x = self.dropout1(x)
#
#         x = self.Conv3(x)
#         x = self.relu3(x)
#         x = self.Conv4(x)
#         x = self.relu4(x)
#         x = self.maxpooling2(x)
#         x = self.dropout2(x)
#
#         x = self.Flatten(x)
#         x = self.dense1(x)
#         x = self.dropout3(x)
#         x = self.dense2(x)
#         return x
#
# model= mymodel()
#
#
# #  使用RMSprop优化器
# opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#
# model.compile(loss='categorical_crossentropy', # 交叉熵函数
#               optimizer=opt,                   # 使用之前定义的优化器
#               metrics=['accuracy'])
# # 方法三专用
# history = model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(x_test, y_test),
#               shuffle=True)

########################################################################################################################
#  使用RMSprop优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', # 交叉熵函数
              optimizer=opt,                   # 使用之前定义的优化器
              metrics=['accuracy'])
#  ------------------------------------------------数据增强-------------------------------------------#
# 判断是否需要数据增强
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:   这将进行预处理和实时数据扩充
    '''
    Keras的图像生成器ImageDataGenerator。这个生成器有很多操作如翻转、旋转和缩放等，目的是生成更加多
    且不一样的图像数据，这样我们得到的训练模型泛化性更加的好，从而得到的模型更加准确。
    '''
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset  设置数据集上的输入均值为0
        samplewise_center=False,  # set each sample mean to 0              设置每个样本均值为0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset  将输入除以数据集的标准差
        samplewise_std_normalization=False,  # divide each input by its std 将每个输入除以它的标准差
        zca_whitening=False,  # apply ZCA whitening  使用ZCA白化图像
        zca_epsilon=1e-06,  # epsilon for ZCA whitening  为ZCA白化
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180) 在范围(深度，0到180)内随机旋转图像
        # randomly shift images horizontally (fraction of total width)
        # 水平随机移动图像(总宽度的一部分)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        # 垂直随机移动图像(总高度的一部分)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear  设定随机剪切范围
        zoom_range=0.,  # set range for random zoom  设置范围为随机变焦
        channel_shift_range=0.,  # set range for random channel shifts  设置范围的随机通道移位
        # set mode for filling points outside the input boundaries  设置输入边界外的填充点模式
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"   用于fill_mode的值= "常量"
        horizontal_flip=True,  # 随机翻转图片
        vertical_flip=False,   # 随机翻转图片
        # 设置重新调平因子(在任何其他转换之前应用)
        rescale=None,
        # 设置将应用于每个输入的函数
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        # 图像数据格式，可以是“channels_first”，也可以是“channels_last”
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        # 保留用于验证的图像部分(严格在0到1之间)
       # validation_split=0.0
        )

    #  计算特征标准化所需的数量
    # (std, mean, and principal components if ZCA whitening is applied).(标准差，平均值，如果使用ZCA美白的主要成分)。
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().将模型安装到datagen.flow()生成的批上。
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )


#  -------------------------- 保存模型 -------------------------------#
# 打印模型框架
print(model.summary())
# 保存模型
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


#  -------------------------- 训练过程可视化 -------------------------------#

import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])    # 训练准确率
plt.plot(history.history['val_acc'])# 测试准确率
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss']) # 训练损失
plt.plot(history.history['val_loss'])# 测试损失
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()










