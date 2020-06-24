# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月24日
# 内容:迁移学习-图像回归
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取数据及与图像预处理
# 3、迁移学习建模
# 4、训练
# 5、模型可视化与保存模型
# 6、训练过程可视化
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import cv2
import pandas as pd
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers, applications
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
path = 'D:\\northwest\\小组视频\\5单层自编码器'#用的fashion minist 数据集，有十类衣服
f = np.load(path)
X_train = f['x_train']
X_test = f['x_test']
Y_train = f['y_train']
Y_test = f['y_test']
f.close()
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
#vgg16只能识别尺寸大于48*48的彩色图片，而数据集是28*28的灰度图像，所以在将数据集灌入迁移学习模型茜，要对图片数据集进行适当的转换，将图片转换成48*48的彩色图片
X_train = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in X_train]
X_test = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB) for i in X_test]
'''
cv2.resize：将原图放大到48*48
cv2.cvtColor(p1,p2) ：是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_GRAY2RGB 将灰度图片转化为格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片

'''
X_train = np.asarray(X_train)#转为array数组
X_test= np.asarray(X_test)#转为array数组
X_train = X_train.astype("float32")/255.#归一化
X_test = X_test.astype("float32")/255.#归一化
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
#  ---------------------- 3、伪造回归数据 ---------------------------------
'''
这里手写体数据集有10类，假设手写体每一类是一类衣服，他有十个标签，所以可以给这10类假设一个价格，
然后对他进行回归预测，就是伪造回归数据。如何伪造，利用了正态分布给每一类标上价格
正态分布的两个重要指标是均值和标准差，决定了整个正态分布的位置和形状，所以这里利用了它的性质。
提前设定好每类的价格（均值），然后设定一个标准差，通过正态分布堆积生成价格然后赋予给对应类的衣服。
'''
#首先将数据转换成Dataframe来处理。因为它比较简单，他是一个表格型的数据类型，每列值的类型可以不同，是最常用的pandas类型
Y_train_pd = pd.DataFrame(Y_train)
Y_test_pd = pd.DataFrame(Y_test)
Y_train_pd.columns = ['label']
Y_test_pd.columns = ['label']
#设置价格，下面是均值列表
'''
sort 与 sorted 区别：
sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
'''
mean_value_list = [45,57,85,99,125,27,180,152,225,33]
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)],3,size=1))[0]#np.random.normal是正态分布，均值有均值列表给出，方差是3，输出的值放在size的shape里面
    return np.round(price,2)
Y_train_pd = Y_train_pd['label'].apply(setting_clothes_price)
Y_test_pd = Y_test_pd['label'].apply(setting_clothes_price)
print(Y_train_pd.head(5))#打印前5个
print(Y_test_pd.head(5))
#  ---------------------- 3、伪造回归数据 ---------------------------------
#  -----------------------4、数据归一化 -----------------------------------
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()#归一化
min_max_scaler.fit(Y_train_pd)#训练集归一化
Y_train = min_max_scaler.transform(Y_train_pd)[:,1]#归一化之后的数据
min_max_scaler.fit(Y_test_pd)#测试集归一化
Y_test= min_max_scaler.transform(Y_test_pd)[:,1]#归一化之后的数据
print(len(Y_train))
print(len(Y_test))
#  -----------------------4、数据归一化 -----------------------------------
#  -----------------------5、迁移学习建模 ---------------------------------
#使用vgg16模型
base_model = applications.VGG16(include_top=False,weight='imagenet',input_shape=X_train.shape[1:])#第一层指定输入大小
print(X_train.shape[1:])#结果与print(base_model.output)输出结果是一样的,1875.0
'''
与图像分类不同的是，图像分类最后要连接自己建立的全连接层，而图像回归要将vgg16的输出层换成回归的输出层，其余层保持不变，
将图像数据导入模型训练就可以了
'''
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Flatten,Dropout
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))#拉平经过vgg16后，图像参数变为 7*7*512，将其加入到自己的全连接结构中
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('Linear'))
mdel = Model(input=base_model.input,output=model(base_model.output_shape))#将vgg16和自己的两层全连接层相连
for layer in model.layers[:15]:#保证vgg16前15层权值不变，在训练过程中不需训练他们
    layer.trainable = False
batch_size = 32  # 设置批大小为32
epochs = 5
data_augmentation = True  # 使用图像增强
import os
save_dir = os.path.join(os.getcwd(), 'saved_models_keras_transferlearning')  # 保存的模型路径
model_name = 'keras_fashion_transferlearning_trained_model.h5'  # 模型名字
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)  # 使用RMSprop优化器
model.compile(loss='mse',#损失用的是均方误差
                  optimizer=opt,  # 使用之前定义的优化器
                  metrics=['accuracy'])
#  -----------------------5、迁移学习建模 ---------------------------------
#  -----------------------6、训练 -----------------------------------------
#判断是否需要数据增强
from keras.preprocessing.image import ImageDataGenerator
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(X_test, Y_test),
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
    datagen.fit(X_train)
    print(X_train.shape[0]//batch_size)  # 取整
    print(X_train.shape[0]/batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().将模型安装到datagen.flow()生成的批上。
    history = model.fit_generator(datagen.flow(X_train, Y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=(X_test, Y_test),
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )

#  -----------------------6、训练 -----------------------------------------
#  -----------------------7、模型可视化与保存模型 -------------------------
# 打印模型框架
print(model.summary())
# 保存模型
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
#  -----------------------7、模型可视化与保存模型 -------------------------
#  ---------------------- 8、训练过程可视化 -------------------------------
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss']) # 训练损失
plt.plot(history.history['val_loss'])# 测试损失
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
#---------------------- 8、训练过程可视化 - ------------------------------
