# -------------------------------------------------开发者信息----------------------------------------------------------#
# 开发者：高强
# 开发日期：2020年6月1日
# 开发框架：keras
# 温馨提示：
#----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------代码布局-----------------------------------------------------------#
# 1、读取手写体数据及与图像预处理
# 2、构建自编码器模型
# 3、训练
# 4、保存模型及模型可视化
# 5、查看自编码器的压缩效果
# 6、查看自编码器的解码效果
# 7、训练过程可视化
#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------读取手写体数据及与图像预处理------------------------------------------------#
import numpy as np
# 载入数据
path = 'F:\\Keras代码学习\\keras\\keras_datasets\\mnist.npz'
f = np.load(path)
print(f.files) # 查看文件内容 ['x_test', 'x_train', 'y_train', 'y_test']
# 定义训练数据 60000个
x_train = f['x_train']
# 定义测试数据 10000个
x_test = f['x_test']
f.close()

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

#----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------构建卷积自编码器模型------------------------------------------------------#

# 方法一： 函数API式
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import Model
input = Input(shape = (28,28,1))
# encoder
conv1 = Conv2D(16,(3, 3), padding='same',activation = 'relu')(input)  # 1*28*28 --> 16*28*28
pool1 = MaxPooling2D(pool_size = (2,2),padding='same')(conv1)         # 16*28*28 -->16*14*14
conv2 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(pool1)   # 16*14*14 --> 8*14*14
pool2 = MaxPooling2D(pool_size = (2,2),padding='same')(conv2)         # 8*14*14  --> 8*7*7
conv3 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(pool2)   # 8*7*7 --> 8*7*7
pool3 = MaxPooling2D(pool_size = (2,2),padding='same')(conv3)         # 8*7*7  --> 8*4*4
# decoder
conv4 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(pool3)   # 8*4*4 --> 8*4*4
up1 = UpSampling2D((2,2))(conv4)                                      # 8*4*4 -->8*8*8
conv5 = Conv2D(8,(3, 3), padding='same',activation = 'relu')(up1)     # 8*8*8 --> 8*8*8
up2 = UpSampling2D((2,2))(conv5)                                      # 8*8*8  --> 8*16*16
conv6 = Conv2D(16,(3, 3),activation = 'relu')(up2)                    # 8*16*16 -->16*14*14 (not same,padding=0)
up3 = UpSampling2D((2,2))(conv6)                                       # 16*14*14  --> 16*28*28
output = Conv2D(1,(3, 3),padding='same',activation = 'sigmoid')(up3)   # 16*28*28--> 1*28*28

model = Model(inputs=input, outputs=output)

# 方法二：class方法（有点问题）
# import keras
# from keras.layers import Input
# from keras import Model
# input = Input(shape = (28,28,1))
#
# class mymodel(keras.Model):
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.conv1 = keras.layers.Conv2D(16,(3, 3),activation = 'relu', padding='same')
#         self.pool1 = keras.layers.MaxPooling2D(pool_size = (2,2),padding='same')
#         self.conv2 = keras.layers.Conv2D(8,(3, 3),activation = 'relu', padding='same')
#         self.pool2 = keras.layers.MaxPooling2D(pool_size = (2,2),padding='same')
#         self.conv3 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
#         self.pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
#
#         self.conv4 = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
#         self.up1 = keras.layers.UpSampling2D((2,2))
#         self.conv5 = keras.layers.Conv2D(8,(3, 3),activation = 'relu', padding='same')
#         self.up2 = keras.layers.UpSampling2D((2, 2))
#         self.conv6 = keras.layers.Conv2D(16,(3, 3),activation = 'relu')
#         self.up3 = keras.layers.UpSampling2D((2,2))
#         self.output = keras.layers.Conv2D(1,(3, 3),activation = 'sigmoid',padding='same')
#
#     def call(self,inputs):
#         conv1 =self.conv1(input)
#         pool1 = self.pool1(conv1)
#         conv2 = self.conv2(pool1)
#         pool2 = self.pool2(conv2)
#         conv3 = self.conv3(pool2)
#         pool3= self.pool3(conv3)
#
#         conv4 = self.conv4(pool3)
#         up1 = self.up1(conv4)
#         conv5 = self.conv5(up1)
#         up2 = self.up1(conv5)
#         conv6 = self.conv6(up2)
#         up3 = self.up1(conv6)
#         output = self.output(up3)
#
#         return output
# model= mymodel()


model.compile(
    optimizer= 'adadelta',
    loss= 'binary_crossentropy'
)
history = model.fit(x_train, x_train,
                    batch_size=128,
                    epochs=3,
                    verbose=2,
                    validation_data=(x_test,x_test)
)
#----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------保存模型、模型可视化-------------------------------------------------------#
from keras.utils import plot_model
# 保存模型
model.save('keras_model_CNNAutoEnconder.h5')
# 模型可视化
plot_model(model, to_file='keras_model_CNNAutoEnconder.png', show_shapes=True)
#----------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------查看自编码器的解码效果---------------------------------------------------#
# decoder做测试
decoder_imgs = model.predict(x_test)
# 打印10张测试集手写数字的压缩效果
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20,6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))        # 将原图转化为 28*28的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 打印解码图
    x = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoder_imgs[i].reshape(28, 28))  # 将解码图转化为 28*28的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------训练过程可视化--------------------------------------------------------#
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
#----------------------------------------------------------------------------------------------------------------------#