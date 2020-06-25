# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/23
#  @Author: MiJizong
#  @Content: 图像回归——Keras
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、伪造回归数据
# 4、数据归一化
# 5、迁移学习建模
# 6、训练
# 7、模型可视化与保存模型
# 8、训练过程可视化
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import cv2
import keras
import numpy as np
import pandas as pd
from keras import Sequential, applications, Input
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
import os
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1个GPU
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
x_train = f['x_train']
y_train = f['y_train']
# 测试数据
x_test = f['x_test']
y_test = f['y_test']
f.close()
# 数据放到本地路径test

# 观察下x_train和x_test维度
print(x_train.shape)  # 输出x_train维度  (60000, 28, 28)
print(x_test.shape)  # 输出x_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    x_train shape: (60000, 48, 48, 3)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# # 数据准备
# # np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# # len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# # X_train 60000*784, X_test10000*784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

batch_size = 32
epochs = 5
data_augmentation = True  # 图像增强
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'keras_fashion_transfer_learning_trained_model.h5'

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、伪造回归数据 ---------------------

# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

# 给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]  # 均值列表


def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]  # 均值mean,标准差std,数量
    return np.round(price, 2)  # 四舍五入保留两位小数


y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

#  --------------------- 3、伪造回归数据 ---------------------

#  --------------------- 4、数据归一化 -----------------------

# 由于mnist的输入数据维度是(num, 28, 28)，vgg-16 需要三维图像,因为扩充一下mnist的最后一维
# cv2.resize(i, (48, 48)) 将原图i转换为48*48
# cv2.COLOR_GRAY2RGB 灰度图转换为RGB图像
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

# 转换为array存储
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

# y_train_price_pd = y_train_pd['price'].tolist()
# y_test_price_pd = y_test_pd['price'].tolist()
# 训练集归一化
min_max_scaler = MinMaxScaler()  # 将属性缩放到一个指定的最大和最小值（通常是0-1）之间
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]  # 数据标准化

# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))

#  --------------------- 4、数据归一化 ---------------------


#  --------------------- 5、迁移学习建模 ---------------------

# 使用VGG16模型
base_model = applications.VGG16(include_top=False,   # include_top=False 表示 不包含最后的3个全连接层
                                weights='imagenet',  # weights：pre-training on ImageNet
                                input_shape=x_train.shape[1:])  # 第一层需要指出图像的大小

# # path to the model weights files.
# top_model_weights_path = 'bottleneck_fc_model.h5'
print(x_train.shape[1:])

# 建立CNN模型
# ************** Sequential 模型 **************
model1 = Sequential()
print(base_model.output)
model1.add(Flatten(input_shape=base_model.output_shape[1:]))
model1.add(Dense(256, activation='relu'))  # 7 * 7 * 512 --> 256
model1.add(Dropout(0.5))
model1.add(Dense(1))  # 256 --> 1
model1.add(Activation('linear'))  # Dense层如果不指定激活函数则会默认activation = ‘linear’

# add the model on top of the convolutional base
# 输入为VGG16的数据，经过VGG16的特征层，2层全连接到1输出（自己加的）
model1 = Model(inputs=base_model.input, outputs=model1(base_model.output))  # VGG16模型与自己构建的模型合并
# ************** Sequential 模型 **************
# ***************** API 模型 ******************
x = Flatten(input_shape=base_model.output_shape[1:])(base_model.output)
x = Dense(256,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1,activation='linear')(x)  # Dense层如果不指定激活函数则会默认activation = ‘linear’
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
        self.dense2 = keras.layers.Dense(1)
    def call(self,input3):
        x = self.flatten(input3)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)
model3 = TL()
# ***************class继承 模型 ****************


# 保持VGG16的前15层权值不变，即在训练过程中不训练
for layer in model1.layers[:15]:
    layer.trainable = False

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model1.compile(loss='mse',
              optimizer=opt,
              )

#  --------------------- 5、迁移学习建模 ---------------------


#  --------------------- 6、训练 ---------------------

# 是否使用数据增强
if not data_augmentation:
    print('Not using data augmentation.')
    history = model1.fit(x_train, y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(x_test, y_test),
                         shuffle=True)      # 表示是否在训练过程中随机打乱输入样本的顺序
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
        rescale=None,                               # 设置缩放比例因子（在进行任何其他转换之前应用）
        preprocessing_function=None,                # 设置将应用于每个输入的函数
        data_format=None,                           # 图像数据格式，“ channels_first”或“ channels_last”
        validation_split=0.0)                       # 保留用于验证的图像比例（严格控制在0和1之间）

    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)  # 取整
    print(x_train.shape[0] / batch_size)   # 保留小数
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

#  --------------------- 6、训练 -----------------------------------


#  --------------------- 7、模型可视化与保存模型 ---------------------

model1.summary()  # 输出模型各层的参数状况
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model1.save(model_path)
print('Saved trained model at %s ' % model_path)

#  --------------------- 7、模型可视化与保存模型 ---------------------


#  --------------------- 8、训练过程可视化 ---------------------------

import matplotlib.pyplot as plt

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#  --------------------- 8、训练过程可视化 ---------------------------
