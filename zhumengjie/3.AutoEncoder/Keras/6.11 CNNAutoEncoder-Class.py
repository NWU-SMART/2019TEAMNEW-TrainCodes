# 开发者：朱梦婕
# 开发日期：2020年6月11日
# 开发框架：keras
#----------------------------------------------------------#
# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy和os的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看解码效果
# 7、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'E:\\study\\kedata\\mnist.npz'
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


#  --------------------- 3、构建卷积自编码器模型 ---------------------

# 输入维度为 1*28*28
inputs = Input(shape=(28, 28,1))

class CNNAutoEncoder(keras.Model):
    def __init__(self):
        super(CNNAutoEncoder, self).__init__(name='CNNAutoEncoder_C')
        self.conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')  # 1*28*28 --> 16*28*28
        self.pool1 = MaxPooling2D((2, 2), padding='same')  # 16*28*28 --> 16*14*14
        self.conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same') # 16*14*14 --> 8*14*14
        self.pool2 = MaxPooling2D((2, 2), padding='same')  # 8*14*14 --> 8*7*7
        self.conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')  # 8*7*7 --> 8*7*7
        self.pool3 = MaxPooling2D((2, 2), padding='same')  # 8*7*7 --> 8*4*4

        self.conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')  # 8*4*4 --> 8*4*4
        self.up1 = UpSampling2D((2, 2))  # 8*4*4 --> 8*8*8
        self.conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')  # 8*8*8 --> 8*8*8
        self.up2 = UpSampling2D((2, 2))  # 8*8*8 --> 8*16*16
        self.conv2_3 = Conv2D(16, (3, 3), activation='relu')  # 8*16*16 --> 16*14*14 (not same)
        self.up3 = UpSampling2D((2, 2)) # 16*14*14 --> 16*28*28
        self.conv2_4 = Conv2D(1, (3, 3), activation='sigmoid', padding='same') # 16*28*28 --> 1*8*8

    def call(self, inputs, mask=None):
        # 编码器
        x = self.conv1_1(inputs)
        x = self.pool1(x)
        x = self.conv1_2(x)
        x = self.pool2(x)
        x = self.conv1_3(x)
        x = self.pool3(x)

        # 解码器
        x = self.conv2_1(x)
        x = self.up1(x)
        x = self.conv2_2(x)
        x = self.up2(x)
        x = self.conv2_3(x)
        x = self.up3(x)
        x = self.conv2_4(x)
        return x

model = CNNAutoEncoder()
model.compile(optimizer='adadelta', loss='binary_crossentropy')

#  --------------------- 3、构建卷积自编码器模型 ---------------------

#  --------------------- 4、模型可视化 ---------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

#  --------------------- 4、模型可视化 ---------------------

#  --------------------- 5、训练 ---------------------

# 设定peochs和batch_size大小
epochs = 3
batch_size = 128

history = model.fit(X_train, X_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(X_test, X_test)
                         )

#  --------------------- 5、训练 ---------------------

#  --------------------- 6、查看解码效果 ---------------------

# decoded_imgs 为输出层的结果
decoded_imgs = model.predict(X_test)

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

#  --------------------- 6、查看解码效果 ---------------------


#  --------------------- 7、训练过程可视化 ---------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------- 7、训练过程可视化 ---------------------