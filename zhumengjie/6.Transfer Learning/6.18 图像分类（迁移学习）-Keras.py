#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月18日
# 开发框架：keras
# 开发内容：图像分类（迁移学习）三种方法
#----------------------------------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, gzip，os 和 cv2的包
# 2、读取数据及与图像预处理
# 3、参数定义
# 4、迁移学习建模
# 5、训练
# 6、模型可视化与保存模型
# 7、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import applications
import cv2
import functools
from keras.models import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第2个GPU
#  -------------------------- 导入需要包 -------------------------------


#  --------------------- 2、读取数据及与图像预处理 ---------------------

# path = 'D:\\keras_datasets\\'

# 函数：数据加载
def load_data():
    paths = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    # 将文件解压并划分为数据集
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

(x_train, y_train), (x_test, y_test) = load_data()     # 加载数据集

# 将类型信息进行one-hot编码(10类)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 由于mist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,因为扩充一下mnist的最后一维
'''
cv2.resize：将原图放大到48*48
cv2.cvtColor(p1,p2) ：是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_GRAY2RGB 将灰度图片转化为格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
'''
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化


#  --------------------- 2、读取数据及与图像预处理 ---------------------

#  ---------------------------------3、参数定义 --------------------------------

batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')  # 保存的模型路径
model_name = 'keras_fashion_transfer_learning_trained_model.h5'         # 保存模型的名字

#  ----------------------------------- 参数定义---------------------------------------------

#  --------------------- 4、迁移学习建模 ---------------------

# 使用VGG16模型
'''
# 关于vgg16:
VGG16共包含：
13个卷积层（Convolutional Layer），分别用conv3-XXX表示
3个全连接层（Fully connected Layer）,分别用FC-XXXX表示
5个池化层（Pool layer）,分别用maxpool表示
其中，卷积层和全连接层具有权重系数，因此也被称为权重层，总数目为13+3=16，这即是VGG16中16的来源。(池化层不涉及权重，因此
不属于权重层，不被计数)。
优缺点：
VGG16的参数数目非常大，可以预期它具有很高的拟合能力；但同时缺点也很明显：即训练时间过长，调参难度大。需要的存储容量大，
不利于部署。例如存储VGG16权重值文件的大小为500多MB，不利于安装到嵌入式系统中。
'''
base_model = applications.VGG16(include_top = False,   # include_top=False 表示 不包含最后的3个全连接层
                                weights = 'imagenet',  # weights: None 代表随机初始化， 'imagenet' 代表加载在 ImageNet 上预训练的权值。
                                input_shape = x_train.shape[1:] # 第一层需要指出图像的大小，input_shape: 可选，输入尺寸元组，仅当 include_top=False 时有效，否则输入形状必须是 (244, 244, 3)（对于 channels_last
                                                                # 数据格式），或者 (3, 244, 244)（对于 channels_first 数据格式）。它必须拥有 3 个输入通道，且宽高必须不小于 32。
                                                                # 例如 (200, 200, 3) 是一个合法的输入尺寸。
                                )


# 建立CNN模型
#  API方法
print(base_model.output)
x = Flatten(input_shape=base_model.output_shape[1:])(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)  # VGG16模型与自己构建的模型合并

'''  class方法：存在错误
class CNNmodel(base_model):
    def __init__(self):
        super(CNNmodel, self).__init__(name='CNN')
        self.base = base_model
        self.flatten = keras.layers.Flatten(input_shape=base_model.output_shape[1:])
        self.dense1 = keras.layers.Dense(256)
        self.relu = keras.layers.ReLU()
        self.dropout = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(num_classes)
        self.softmax = keras.layers.Softmax()

    def call(self, inputs):
        x = self.base(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x
model = CNNmodel()
'''

''' Sequential方法：
model = Sequential()
print(base_model.output)  # 1875.0
model.add(Flatten(input_shape=base_model.output_shape[1:]))  # 展开

# 经过vgg16后，图像参数变为 7*7*512，将其加入到我们自己的全连接结构中
model.add(Dense(256, activation='relu'))  # 7 * 7 * 512 --> 256
model.add(Dropout(0.5))

model.add(Dense(num_classes))  # 256 --> num_classes
model.add(Activation('softmax'))

# add the model on top of the convolutional base
# 输入为VGG16的数据，经过VGG16的特征层，2层全连接到num_classes输出（自己加的）
model = Model(inputs=base_model.input, outputs=model(base_model.output))  # VGG16模型与自己构建的模型合并
'''
# 保持VGG16的前15层权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False

# 初始化 RMSprop 优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',  # 交叉熵
              optimizer=opt,                    # 优化器
              metrics=['accuracy'])

#  --------------------- ---------迁移学习建模 ----------------------------


#  --------------------- 5、训练 ---------------------
# 判断是否需要数据增强：
# 不需要：
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
# 需要：
else:
    print('Using real-time data augmentation.')
    # 预处理和实时数据扩充：
    '''
        Keras的图像生成器ImageDataGenerator。这个生成器有很多操作如翻转、旋转和缩放等，目的是生成更加多
        且不一样的图像数据，这样我们得到的训练模型泛化性更加的好，从而得到的模型更加准确。
        '''
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 将输入数据的均值设为0
        samplewise_center=False,  # 将每个样本的均值设为0
        featurewise_std_normalization=False,  # 逐个特征将输入数据除以标准差
        samplewise_std_normalization=False,  # 将每个输入除以标准差
        zca_whitening=False,  # 是否使用zca白化（降低输入的冗余性）
        zca_epsilon=1e-06,  # 利用阈值构建低通滤波器对输入数据进行滤波
        rotation_range=0,  # 随机旋转的度数
        width_shift_range=0.1,  # 水平随机移动宽度的0.1
        height_shift_range=0.1,  # 垂直随机移动高度的0.1
        shear_range=0.,  # 不随机剪切
        zoom_range=0.,  # 不随机缩放
        channel_shift_range=0.,  # 通道不随机转换
        fill_mode='nearest',  # 边界以外的点的填充模式：aaaaaaaa|abcd|dddddddd靠近哪个就用哪个填充，如靠近a那么就用a填充，靠近d就用d填充
        cval=0.,  # 边界之外点的值
        horizontal_flip=True,  # 随机水平翻转
        vertical_flip=False,  # 不随机垂直翻转
        rescale=None,  # 不进行缩放（若不为0和None将数据乘以所提供的值）
        preprocessing_function=None,  # 应用于输入的函数
        data_format=None,  # 图像数据格式
        validation_split=0.0)  # 用于验证图像的比例

    #  计算特征标准化所需的数量
    # (std, mean, and principal components if ZCA whitening is applied).(标准差，平均值，如果使用ZCA美白的主要成分)。
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # 将模型安装到datagen.flow()生成的批上。
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )


#  ---------------------训练 ---------------------



#  --------------------- 6、保存模型 ---------------------

model.summary()
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#  --------------------- 保存模型 ---------------------

#  --------------------- 7、训练过程可视化 ---------------------

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

#  ---------------------训练过程可视化 ---------------------



