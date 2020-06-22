# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月22日
# 内容:迁移学习-图像分类
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
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras import applications
import cv2
import functools
from keras.models import load_model
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
path = 'D:\\northwest\\小组视频\\5单层自编码器\\mnist.npz'
def load_data():
    paths = [
        path+'train-labels-idx1-ubyte.gz',path+'train-images-idx3-ubyte.gz',
        path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0],'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)
    with gzip.open(paths[1],'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(),np.uint8,offset=8).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2],'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(),np.uint8,offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=8).reshape(len(y_test), 28, 28, 1)
    return(x_train,y_train),(x_test,y_test)
#读数据
(x_train,y_train),(x_test,y_test)=load_data()
batch_size = 128
num_calsses = 10
epochs = 5
data_augmentation = True#数据增强
num_preditions = 20
save_dir = os.path.join(os.getcwd(),'saved_model_transfer_learning')
model_name='keras_fashion_transfer_learning_trained_h5'
y_train=keras.utils.to_categorical(y_train,num_calsses)
y_test=keras.utils.to_categorical(y_test,num_calsses)
#minist图像输入维度是二维的，但是这里用的VGG16，需要三维数据，所以扩充它的最后一维
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
x_train = np.asarray(X_train)
x_test = np.asarray(X_test)
x_train = x_train.astype('folat32')/255
x_test = x_test.astype('folat32')/255
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
#  ---------------------- 3、迁移学习建模-----------------------------------
#基准模型使用的是VGG16，因此要去掉最后的三个全连接层，include_top=False,第一层指定它的输入图像的大小
base_model = applications.VGG16(include_top=False,weights='imagenet',input_shape=x_train.shape[1:])
#建立CNN模型
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model,output_shape=x_train.shape[1:]))
model.add(Dense(256),Activation='relu')#7*7*512-256
model.add(Dropout(0.5))
model.add(Dense(num_calsses))
model.add(Activation='softmax')
model = Model(inputs=base_model.input,outputs=base_model.output)## 输入为VGG16的数据，经过VGG16的特征层，2层全连接到num_classes输出，这是自己加的
for layer in model.layer[:15]:
    layer.trainable=False#前15层不训练，保留vgg16前15层权值不变
opt = keras.optimizers.rmsprop(lr=0.0001,decay=1e-6)#初始化rmspro优化器
model.compile(loss='categorical_crossentropy',optimizer=opt)
#  ---------------------- 3、迁移学习建模-----------------------------------
#  ---------------------- 4、模型训练---------------------------------------
if not data_augmentation:
  print('Not use data_augmentation')
  history=model.fit(x_train,y_train,batch_size=128,
                    epochs=5,validation_data=(x_test,y_test),
                    shuffle=True)
else:
    print('USE real-time data_augmentation')
    datagen=ImageDataGenerator(
        featurewise_center=False,#输入值按照均值为0进行处理
        samplewise_center=False,#每个样本的均值按0处理
        featurewise_std_normalization=False,#输入值按照标准正态化处理
        samplewise_std_normalization=False,#每个样本按照标准正态化处理
        zca_whitening=False,# 是否开启增白
        zca_epsilon=1e-06,#ZCA使用的eposilon,默认1e-6
        rotation_range=0, #图像随机旋转一定角度，最大旋转角度为设定值
        width_shift_range=0.1,#图像随机水平平移，最大平移值为设定值。若值为小于1的float值，则可认为是按比例平移
        #若大于1，则平移的是像素；若值为整型，平移的也是像素；假设像素为2.0，则移动范围为[-1,1]之间
        height_shift_range=0.1,#图像随机垂直平移
        shear_range=0, # 图像随机修剪
        zoom_range=0,# 图像随机变焦
        channel_shift_range=0, # 浮点数，随机通道偏移的幅度
        fill_mode='nearest',#填充模式，默认为最近原则，比如一张图片向右平移，那么最左侧部分会被临近的图案覆盖 ‘constant’，‘nearest’，‘reflect’或‘wrap’
        cval=0, # 当fill_mode=constant时，指定要向超出边界的点填充的值,浮点数或整数，
        horizontal_flip=True,#图像随机水平翻转，指定为布尔值
        vertical_flip=False, #图像随机垂直翻转
        rescale=None,#缩放尺寸
        preprocessing_function=None,# 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），
        data_format=None,# "channels_first" or "channels_last",如果不指定，则默认是channels_last
        #channels_first=(batch, channels, height, width),channels_last=(batch, height, width, channels)
        validation_split=0.0,)#为验证保留的图像的一部分(严格在0到1之间)

    datagen.fit(x_train)
    print(x_train.shape[0]/batch_size)#取整
    print(x_test.shape[0]/batch_size)
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                                batch_size=128),   # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                                                epochs=5,                                 # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10 ) # 在使用基于进程的线程时，最多需要启动的进程数量
#  ---------------------- 4、模型训练---------------------------------------
#  ---------------------- 5、保存模型---------------------------------------
model.summary()
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
#  ---------------------- 5、保存模型---------------------------------------
#  ---------------------- 6、模型可视化-------------------------------------
import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
'''
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()
'''
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
#  ---------------------- 6、模型可视化-------------------------------------