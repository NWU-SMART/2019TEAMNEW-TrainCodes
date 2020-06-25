#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.25
#software：pycharm
#项目名称：图像回归（keras）
#--------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 8、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import applications
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Flatten,Dropout
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'D:\\keras_datasets\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_']
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))


#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------
#-------------------------------3、建立模型---------------------------------

# 使用vgg16模型
base_model = applications.VGG16(include_top = False,weights = 'imagenet',input_shape = x_train.shape[1:])   # 第一层需要指出图像的大小
print(x_train.shape[1:])
model = Sequential()
print(base_model.output)   # 1875.0
model.add(Flatten(input_shape = base_model.output_shape[1:]))      # 拉平
model.add(Dense(256))    # 经过vgg16后，图像参数变为 7*7*512，将其加入到我们自己的全连接结构中
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))     # 输出层是1类
model.add(Activation('linear'))
model = Model(input=base_model.input,outputs= model(base_model.output))    # 合并vgg16和自己的两层全连接
for layer in model.layers[:15]:    # 保持搭建的网络的前15层（共15层）权值不变，即在训练过程中不训练
    layer.trainable = False

#-------------------------------3、建立模型---------------------------------

#  ---------------------- 4、模型训练---------------------------------------
if not data_augmentation:
  print('Not use data_augmentation')
  history=model.fit(x_train,y_train,batch_size=128,
                    epochs=5,validation_data=(x_test,y_test),
                    shuffle=True)
else:
    print('USE real-time data_augmentation')
    datagen=ImageDataGenerator(
        featurewise_center=False,                      # 输入值按照均值为0进行处理
        samplewise_center=False,                       # 每个样本的均值按0处理
        featurewise_std_normalization=False,           # 输入值按照标准正态化处理
        samplewise_std_normalization=False,            # 每个样本按照标准正态化处理
        zca_whitening=False,                           # 是否开启增白
        zca_epsilon=1e-06,                             # ZCA使用的eposilon,默认1e-6
        rotation_range=0,                              # 图像随机旋转一定角度，最大旋转角度为设定值
        width_shift_range=0.1,                         # 图像随机水平平移，最大平移值为设定值。若值为小于1的float值，则可认为是按比例平移
        # 若大于1，则平移的是像素；若值为整型，平移的也是像素；假设像素为2.0，则移动范围为[-1,1]之间
        height_shift_range=0.1,                        # 图像随机垂直平移
        shear_range=0,                                 # 图像随机修剪
        zoom_range=0,                                  # 图像随机变焦
        channel_shift_range=0,                         # 浮点数，随机通道偏移的幅度
        fill_mode='nearest',                           # 填充模式，默认为最近原则，比如一张图片向右平移，那么最左侧部分会被临近的图案覆盖 ‘constant’，‘nearest’，‘reflect’或‘wrap’
        cval=0,                                        # 当fill_mode=constant时，指定要向超出边界的点填充的值,浮点数或整数，
        horizontal_flip=True,                          # 图像随机水平翻转，指定为布尔值
        vertical_flip=False,                           # 图像随机垂直翻转
        rescale=None,                                  # 缩放尺寸
        preprocessing_function=None,                   # 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），
        data_format=None,                              # "channels_first" or "channels_last",如果不指定，则默认是channels_last
        # channels_first=(batch, channels, height, width),channels_last=(batch, height, width, channels)
        validation_split=0.0,)                         # 为验证保留的图像的一部分(严格在0到1之间)

    datagen.fit(x_train)
    print(x_train.shape[0]/batch_size)                 # 取整
    print(x_test.shape[0]/batch_size)
    history = model.fit_generator(datagen.flow(x_train, y_train,    # 按batch_size大小从x,y生成增强数据
                                                batch_size=128),    # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                                                epochs=5,           # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                                                steps_per_epoch=x_train.shape[0] // batch_size,
                                                validation_data=(x_test, y_test),
                                                workers=10 )        # 在使用基于进程的线程时，最多需要启动的进程数量
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