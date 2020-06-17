# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/15
#  @Author: MiJizong
#  @Content: 去噪自编码器——Keras三种方法实现
#  @Version: 1.0
#  @FileName: 1.0.py 
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： --------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、训练
# 5、查看解码效果
# 6、训练过程可视化
# ----------------------   代码布局： --------------------------------------

#  -------------------------- 1、导入需要包 --------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
'''不加下面这几句，则CONV 报错
tensorflow.python.framework.errors_impl.UnknownError:  Failed to get convolution algorithm. 
This is probably because cuDNN failed to initialize, so try looking to see if a warning log 
message was printed above.
	 [[node conv2d_1/convolution (defined at \ProgramData\Anaconda3x\envs\t2.0\lib\site-packages
	 \tensorflow_core\python\framework\ops.py:1751) ]] [Op:__inference_keras_scratch_graph_1977]
Function call stack:
keras_scratch_graph'''
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#  -------------------------- 1、导入需要包 --------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train = f['x_train']
# 测试数据
X_test = f['x_test']
f.close()
# 数据放到本地路径

# 数据格式进行转换
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

# 加入噪声数据
noise_factor = 0.5  # 噪声因子
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
# 正态分布的均值(0.0表示以x=0为对称轴)、标准差(越小越矮胖)和shape

X_train_noisy = np.clip(X_train_noisy, 0., 1.)  # 截取函数，使用0代替数组中小于0数 使用1代替数组中大于1的数
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

#  --------------------- 2、读取手写体数据及与图像预处理 ----------------------

#  --------------------- 3.1、构建去噪自编码器Sequential模型 -----------------
autoencoder1 = Sequential()

#编码器
autoencoder1.add(Conv2D(32,(3,3),padding='same',activation='relu'))             # 28*28*1 --> 28*28*32
autoencoder1.add(MaxPooling2D((2,2),padding='same'))                            # 28*28*32 --> 14*14*32
autoencoder1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))         # 14*14*32 --> 14*14*32
autoencoder1.add(MaxPooling2D((2, 2), padding='same'))                          # 14*14*32 --> 7*7*32

#解码器
autoencoder1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))         # 7*7*32 --> 7*7*32
autoencoder1.add(UpSampling2D((2, 2)))                                          # 7*7*32 --> 14*14*32
autoencoder1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))         # 14*14*32 --> 14*14*32
autoencoder1.add(UpSampling2D((2, 2)))                                          # 14*14*32 --> 28*28*32
autoencoder1.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))       # 28*28*32 --> 28*28*1

#  --------------------- 3.1、构建去噪自编码器Sequential模型 -----------------

#  --------------------- 3.2、构建去噪自编码器API方法模型 --------------------
# 输入维度为 1*28*28
x = Input(shape=(28, 28, 1))

# 编码器
conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)      # 28*28*1 --> 28*28*32
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)                   # 28*28*32 --> 14*14*32
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)  # 14*14*32 --> 14*14*32
h = MaxPooling2D((2, 2), padding='same')(conv1_2)                       # 14*14*32 --> 7*7*32


# 解码器
conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)      # 7*7*32 --> 7*7*32
up1 = UpSampling2D((2, 2))(conv2_1)                                     # 7*7*32 --> 14*14*32
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)    # 14*14*32 --> 14*14*32
up2 = UpSampling2D((2, 2))(conv2_2)                                     # 14*14*32 --> 28*28*32
r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)        # 28*28*32 --> 28*28*1

autoencoder2 = Model(inputs=x, outputs=r)


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(autoencoder2).create(prog='dot', format='svg'))

#  --------------------- 3.2、构建去噪自编码器API方法模型 --------------------

#  --------------------- 3.3、构建去噪自编码器class继承模型 ------------------
# 输入维度为 1*28*28
inputs3 = Input(shape=(28, 28,1))

class Autoencoder(keras.Model):
    def __init__(self):
        super(Autoencoder,self).__init__()
        #编码器
        self.conv1 = keras.layers.Conv2D(32, (3, 3),padding='same',activation='relu')  # 28*28*1 --> 28*28*32
        self.maxpool1 = keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same')     # 28*28*32 --> 14*14*32
        self.conv2 = keras.layers.Conv2D(32, (3, 3),activation='relu',padding='same')  # 14*14*32 --> 14*14*32
        self.maxpool2 = keras.layers.MaxPooling2D(pool_size=(2, 2),padding='same')     # 14*14*32 --> 7*7*32

        #解码器
        self.conv3 = keras.layers.Conv2D(32, (3, 3),activation='relu',padding='same')    # 7*7*32 --> 7*7*32
        self.upsamp1 = keras.layers.UpSampling2D((2,2))                                  # 7*7*32 --> 14*14*32
        self.conv4 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')  # 14*14*32 --> 14*14*32
        self.upsamp2 = keras.layers.UpSampling2D((2, 2))                                 # 14*14*32 --> 28*28*32
        self.conv5 = keras.layers.Conv2D(1, (3, 3), activation='relu', padding='same')   # 28*28*32 --> 28*28*1

    def call(self,input3):
        x = self.conv1(input3)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.upsamp1(x)
        x = self.conv4(x)
        x = self.upsamp2(x)
        x = self.conv5(x)
        return x

autoencoder3 = Autoencoder()
#  --------------------- 3.3、构建去噪自编码器class继承模型 ------------------

#  --------------------------- 4、训练 -------------------------------------

epochs = 3
batch_size = 128

autoencoder3.compile(optimizer='adadelta', loss='binary_crossentropy')  # AdaDelta算法优化，二分类的交叉熵做loss

history = autoencoder3.fit(X_train_noisy, X_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_test_noisy, X_test))

#  --------------------------- 4、训练 --------------------------------------


#  ------------------------ 5、查看解码效果 ----------------------------------

# decoded_imgs 为输出层的结果
decoded_imgs = autoencoder3.predict(X_test_noisy)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
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

#  --------------------- 5、查看解码效果 ------------------------------------


#  --------------------- 6、训练过程可视化 ----------------------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------- 6、训练过程可视化 -----------------------------------