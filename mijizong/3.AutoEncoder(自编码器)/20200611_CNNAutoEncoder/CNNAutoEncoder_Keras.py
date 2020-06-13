# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/11
#  @Author: MiJizong
#  @Content: 卷积自编码器——Keras三种方法实现
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入相关的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器三种模型
# 4、模型可视化
# 5、训练
# 6、查看解码效果
# 7、训练过程可视化
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
plt.style.use('ggplot')
'''不加这几句，则CONV 报错
tensorflow.python.framework.errors_impl.UnknownError:  Failed to get convolution algorithm. 
This is probably because cuDNN failed to initialize, so try looking to see if a warning log 
message was printed above.
	 [[node conv2d_1/convolution (defined at \ProgramData\Anaconda3x\envs\t2.0\lib\site-packages
	 \tensorflow_core\python\framework\ops.py:1751) ]] [Op:__inference_keras_scratch_graph_1977]
Function call stack:
keras_scratch_graph'''
import  tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 数据放到本地路径

# 数据格式进行转换
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#  数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
# 输出X_train和X_test维度
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28, 1)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3.1、构建卷积自编码器Sequential模型 ----------------
autoencoder1 = Sequential()

#编码器
autoencoder1.add(Conv2D(16,(3,3),padding='same',activation='relu'))             # 1*28*28 --> 16*28*28
autoencoder1.add(MaxPooling2D((2,2),padding='same'))                            # 16*28*28 --> 16*14*14
autoencoder1.add(Conv2D(8, (3, 3), activation='relu', padding='same'))          # 16*14*14 --> 8*14*14
autoencoder1.add(MaxPooling2D((2, 2), padding='same'))                          # 8*14*14 --> 8*7*7
autoencoder1.add(Conv2D(8, (3, 3), activation='relu', padding='same'))          # 8*7*7 --> 8*7*7
autoencoder1.add(MaxPooling2D((2, 2), padding='same'))                          # 8*7*7 --> 8*4*4

#解码器
autoencoder1.add(Conv2D(8, (3, 3), activation='relu', padding='same'))          # 8*4*4 --> 8*4*4
autoencoder1.add(UpSampling2D((2, 2)))                                          # 8*4*4 --> 8*8*8
autoencoder1.add(Conv2D(8, (3, 3), activation='relu', padding='same'))          # 8*8*8 --> 8*8*8
autoencoder1.add(UpSampling2D((2, 2)))                                          # 8*8*8 --> 8*16*16
autoencoder1.add(Conv2D(16, (3, 3), activation='relu'))                         # 8*16*16 --> 16*14*14 (not same)
autoencoder1.add(UpSampling2D((2, 2)))                                          # 16*14*14 --> 16*28*28
autoencoder1.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))       # 16*28*28 --> 1*28*28

#  --------------------- 3.1、构建多层自编码器Sequential模型 ----------------

#  --------------------- 3.2、构建卷积自编码器API模型 -----------------------

# 输入维度为 1*28*28
x = Input(shape=(28, 28,1))

# 编码器
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)      # 1*28*28 --> 16*28*28   16为滤波器的数目（即输出的维度16卷积核）大小3*3 "SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)                   # 16*28*28 --> 16*14*14
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)   # 16*14*14 --> 8*14*14
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)                   # 8*14*14 --> 8*7*7
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)   # 8*7*7 --> 8*7*7
h = MaxPooling2D((2, 2), padding='same')(conv1_3)                       # 8*7*7 --> 8*4*4


# 解码器
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(h)       # 8*4*4 --> 8*4*4
up1 = UpSampling2D((2, 2))(conv2_1)                                     # 8*4*4 --> 8*8*8
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)     # 8*8*8 --> 8*8*8
up2 = UpSampling2D((2, 2))(conv2_2)                                     # 8*8*8 --> 8*16*16
conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)                    # 8*16*16 --> 16*14*14 (not same)
up3 = UpSampling2D((2, 2))(conv2_3)                                     # 16*14*14 --> 16*28*28
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up3)        # 16*28*28 --> 1*28*28

autoencoder2 = Model(inputs=x, outputs=r)

#  --------------------- 3.2、构建卷积自编码器API模型 -----------------------

#  --------------------- 3.3、构建卷积自编码器class继承模型 -----------------
# 输入维度为 1*28*28
inputs = Input(shape=(28, 28,1))

class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder,self).__init__()
        #编码器
        self.conv1 = keras.layers.Conv2D(16, (3, 3),padding='same',activation='relu')
        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same')
        self.conv2 = keras.layers.Conv2D(8, (3, 3),activation='relu',padding='same')
        self.maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same')
        self.conv3 = keras.layers.Conv2D(8, (3, 3),activation='relu', padding='same')
        self.maxpool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')

        #解码器
        self.conv4 = keras.layers.Conv2D(8, (3, 3),activation='relu',padding='same')
        self.upsamp1 = keras.layers.UpSampling2D((2,2))
        self.conv5 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.upsamp2 = keras.layers.UpSampling2D((2, 2))
        self.conv6 = keras.layers.Conv2D(16, (3, 3), activation='relu')
        self.upsamp3 = keras.layers.UpSampling2D((2, 2))
        self.conv7 = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')

    def call(self,input3):
        x = self.conv1(input3)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.upsamp1(x)
        x = self.conv5(x)
        x = self.upsamp2(x)
        x = self.conv6(x)
        x = self.upsamp3(x)
        x = self.conv7(x)
        return x

autoencoder3 = Autoencoder()
#  --------------------- 3.3、构建卷积自编码器class继承模型 -----------------


#  --------------------- 4、模型可视化 -------------------------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(autoencoder1).create(prog='dot', format='svg'))

#  --------------------- 4、模型可视化 -------------------------------------

#  --------------------- 5、训练 -------------------------------------------

# 设定peochs和batch_size大小
epochs = 3
batch_size = 128

autoencoder1.compile(optimizer='adadelta',loss='binary_crossentropy')

history = autoencoder1.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_test, X_test)
                         )

#  --------------------- 5、训练 -------------------------------------------

#  --------------------- 6、查看解码效果 ------------------------------------

# decoded_imgs 为输出层的结果
decoded_imgs = autoencoder1.predict(X_test)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 解码效果图
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

#  --------------------- 6、查看解码效果 ------------------------------------


#  --------------------- 7、训练过程可视化 ----------------------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------- 7、训练过程可视化 ----------------------------------