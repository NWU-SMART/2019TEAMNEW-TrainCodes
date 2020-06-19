#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月19日
# 开发框架：keras
# 开发内容：图像回归（迁移学习）
#----------------------------------------------------------#
'''
源程序存在错误：1、缺少包未导入：import pandas as pd
                               from sklearn.preprocessing import MinMaxScaler
                               from keras import applications
                               from keras.models import Sequential, Model
                               from keras.preprocessing.image import ImageDataGenerator
                               import cv2
                2、vgg16 需要三维图像,未扩充mnist的维度：
                                # 扩充mnist最后一维
                                X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
                                X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
                                x_train = np.asarray(X_train)
                                x_test = np.asarray(X_test)
                3、未导入数据标签：y_train = f['y_train']   # 加载label
                                  y_test = f['y_test']    # 加载label
                4、x_train x_test 程序前后x大小写不一致
                5、部分参数未定义：batch_size = 32
                                  epochs = 5
                                  data_augmentation = True  # 图像增强
                                  save_dir = os.path.join(os.getcwd(), 'saved_models2_transfer_learning')  # 保存的模型路径
                                  model_name = 'keras_fashion_transfer_learning_trained_model2.h5'         # 保存模型的名字
程序在服务器跑的，使用第二块gpu
'''

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, pandas, numpy, sklearn，os 和 cv2的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、参数定义
# 6、训练
# 7、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 9、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import applications
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'  # 第二块
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
x_train = f['x_train']
y_train = f['y_train']   # 加载label
# 测试数据
x_test = f['x_test']
y_test = f['y_test']    # 加载label
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print(x_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(x_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


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
    price = sorted(np.random.normal(mean_value_list[int(row)], 3,size=1))[0] #均值mean,标准差std,数量
    return np.round(price, 2)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

#  --------------------- 3、伪造回归数据 ---------------------

#  --------------------- 4、数据归一化 ---------------------

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

# y_train_price_pd = y_train_pd['price'].tolist()
# y_test_price_pd = y_test_pd['price'].tolist()
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))

#  --------------------- 4、数据归一化 ---------------------

#  --------------------- 5、参数定义 ---------------------
batch_size = 32
epochs = 5
data_augmentation = True  # 图像增强
save_dir = os.path.join(os.getcwd(), 'saved_models2_transfer_learning')  # 保存的模型路径
model_name = 'keras_fashion_transfer_learning_trained_model2.h5'         # 保存模型的名字
#  --------------------- 参数定义 ---------------------

#  --------------------- 6、迁移学习建模 ---------------------

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
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])  # 第一层需要指出图像的大小

# 模型建立
print(x_train.shape[1:])
#  API方法
print(base_model.output)
x = Flatten(input_shape=base_model.output_shape[1:])(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='linear')(x)
model = Model(inputs=base_model.input, outputs=x)  # VGG16模型与自己构建的模型合并

''' Sequential方法
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=model(base_model.output))  # VGG16模型与自己构建的模型合并
'''
# 保持VGG16的前15层权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False

# 初始化 RMSprop 优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='mse',
              optimizer=opt,
              )

#  --------------------- 迁移学习建模 ---------------------


#  --------------------- 7、训练 ---------------------
if not data_augmentation:
    print('Not using data augmentation.')
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
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )

#  --------------------- 训练 ---------------------


#  --------------------- 8、模型可视化与保存模型 ---------------------

model.summary()
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#  --------------------- 模型可视化与保存模型 ---------------------


#  --------------------- 9、训练过程可视化 ---------------------

import matplotlib.pyplot as plt
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#  --------------------- 训练过程可视化 ---------------------
