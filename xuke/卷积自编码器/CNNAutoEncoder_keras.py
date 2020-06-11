#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.11
#software：pycharm
#项目名称：CNN自编码器（keras）
#--------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看解码效果
# 7、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import keras
from keras.layers import Input
from keras import Model
from keras.models import Sequential
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 -------------------------------

#-------------------2、读取手写体数据及与图像预处理-----------------------
# 载入数据
path = 'D:\\keras\\编码器\\mnist.npz'           # 数据集路径
f = np.load(path)                              # 打开文件
print(f.files)                                 # 查看文件内容 ['x_test', 'x_train', 'y_train', 'y_test']
x_train = f['x_train']                         # 定义训练数据 60000个
x_test = f['x_test']                           # 定义测试数据 10000个
f.close()                                      # 关闭文件
# 数据预处理
# 数据格式进行转换
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.

# 打印训练数据的维度
print(x_train.shape)  # (60000, 28, 28,1)
# 打印测试数据的维度
print(x_test.shape)   # (10000, 28, 28,1)
##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28, 1)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

#-----------------------2、读取手写体数据及与图像预处理--------------------

#----------------------3、构建卷积自编码器模型----------------------------

#---------------------------3.1 sequential（）--------------------------
CNNmodel = Sequential()
# 编码
CNNmodel.add(Conv2D(16, (3, 3), activation='relu', padding='same'))  # 1*28*28 --> 16*28*28
CNNmodel.add(MaxPooling2D((2, 2), padding='same'))                   # 16*28*28 -->16*14*14
CNNmodel.add(Conv2D(8, (3, 3), activation='relu', padding='same'))   # 16*14*14 --> 8*14*14
CNNmodel.add(MaxPooling2D((2, 2), padding='same'))                   # 8*14*14  --> 8*7*7
CNNmodel.add(Conv2D(8, (3, 3), activation='relu', padding='same'))   # 8*7*7 --> 8*7*7
CNNmodel.add(MaxPooling2D((2, 2), padding='same'))                   # 8*7*7  --> 8*4*4

#解码
CNNmodel.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
CNNmodel.add(UpSampling2D((2, 2)))
CNNmodel.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
CNNmodel.add(UpSampling2D(2, 2))
CNNmodel.add(Conv2D(16, (3, 3), activation='relu'))
CNNmodel.add(UpSampling2D((2, 2)))
CNNmodel.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model = CNNmodel()
model.compile(optimizer='adadelta', loss='binary_crossentropy')
#---------------------------3.1 sequential（）--------------------------

#-------------------------------3.2 API---------------------------------
input = Input(shape = (28,28,1))
# 编码
conv1 = Conv2D(16,(3, 3), padding='same',activation = 'relu')(input)  # 1*28*28 --> 16*28*28
pool1 = MaxPooling2D(pool_size = (2,2),padding='same')(conv1)         # 16*28*28 -->16*14*14
conv2 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(pool1)   # 16*14*14 --> 8*14*14
pool2 = MaxPooling2D(pool_size = (2,2),padding='same')(conv2)         # 8*14*14  --> 8*7*7
conv3 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(pool2)   # 8*7*7 --> 8*7*7
pool3 = MaxPooling2D(pool_size = (2,2),padding='same')(conv3)         # 8*7*7  --> 8*4*4
# 解码
conv4 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(pool3)   # 8*4*4 --> 8*4*4
up1 = UpSampling2D((2,2))(conv4)                                      # 8*4*4 -->8*8*8
conv5 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(up1)     # 8*8*8 --> 8*8*8
up2 = UpSampling2D((2,2))(conv5)                                      # 8*8*8  --> 8*16*16
conv6 = Conv2D(16,(3, 3),activation = 'relu')(up2)                    # 8*16*16 -->16*14*14 (not same,padding=0)
up3 = UpSampling2D((2,2))(conv6)                                       # 16*14*14  --> 16*28*28
output = Conv2D(1,(3, 3),padding='same',activation = 'sigmoid')(up3)   # 16*28*28--> 1*28*28
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adadelta', loss='binary_crossentropy')
#-------------------------------3.2 API---------------------------------

#-------------------------------3.3 class-------------------------------
class CNNmodel(keras.Model):
     def __init__(self):
         super(CNNmodel,self).__init__()
         # 编码
         self.conv1 = keras.layers.Conv2D(16,(3, 3),activation = 'relu', padding='same')
         self.pool1 = keras.layers.MaxPooling2D(pool_size = (2,2),padding='same')
         self.conv2 = keras.layers.Conv2D(8,(3, 3),activation = 'relu', padding='same')
         self.pool2 = keras.layers.MaxPooling2D(pool_size = (2,2),padding='same')
         self.conv3 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
         self.pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        # 解码
         self.conv4 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
         self.up1 = keras.layers.UpSampling2D((2,2))
         self.conv5 = keras.layers.Conv2D(8,(3, 3),activation = 'relu', padding='same')
         self.up2 = keras.layers.UpSampling2D((2, 2))
         self.conv6 = keras.layers.Conv2D(16,(3, 3),activation = 'relu')
         self.up3 = keras.layers.UpSampling2D((2,2))
         self.output = keras.layers.Conv2D(1,(3, 3),activation = 'sigmoid',padding='same')
     def call(self,inputs):                # 前向传播
         conv1 = self.conv1(input)
         pool1 = self.pool1(conv1)
         conv2 = self.conv2(pool1)
         pool2 = self.pool2(conv2)
         conv3 = self.conv3(pool2)
         pool3 = self.pool3(conv3)
         conv4 = self.conv4(pool3)
         up1 = self.up1(conv4)
         conv5 = self.conv5(up1)
         up2 = self.up1(conv5)
         conv6 = self.conv6(up2)
         up3 = self.up1(conv6)
         output = self.output(up3)
         return output
model= CNNmodel()
model.compile(optimizer='adadelta', loss='binary_crossentropy')

#-------------------------------3.2 class-------------------------------

#----------------------3、构建卷积自编码器模型----------------------------

#  --------------------- 4、模型可视化 ---------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(CNNmodel).create(prog='dot', format='svg'))

#  --------------------- 4、模型可视化 ---------------------

#  --------------------- 5、训练 ---------------------

# 设定peochs和batch_size大小
epochs = 3
batch_size = 128
history = CNNmodel.fit(x_train, x_train,batch_size=batch_size,epochs=epochs, verbose=1,validation_data=(x_test, x_test))

#  --------------------- 5、训练 ---------------------

#  --------------------- 6、查看解码效果 ---------------------

# decoded_imgs 为输出层的结果
decoded_imgs = CNNmodel.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
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