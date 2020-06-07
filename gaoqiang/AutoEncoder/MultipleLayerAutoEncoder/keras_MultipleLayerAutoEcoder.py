# -------------------------------------------------开发者信息----------------------------------------------------------#
# 开发者：高强
# 开发日期：2020年5月29日
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
# 打印训练数据的维度
print(x_train.shape)  # (60000, 28, 28)
# 打印测试数据的维度
print(x_test.shape)   # (10000, 28, 28)

# 数据预处理
# 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.
# np.load是将28*28的矩阵转换为1*784，方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))  # 60000*784
x_test  = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))     # 10000*784

#----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------构建多层自编码器模型------------------------------------------------------#
#                                        784->128->64->128->784 (三个隐藏层)                                           #
# 方法一： 函数API式
from keras.layers import Input
from keras.layers import Dense
from keras import Model
input = Input(shape = (784,))
hidden1 = Dense(128,activation = 'relu')(input)          # encoder
hidden2 = Dense(64,activation = 'relu')(hidden1)
hidden3 = Dense(128,activation = 'relu')(hidden2)        # decoder
output = Dense(784,activation = 'sigmoid')(hidden3)

model = Model(inputs=input, outputs=output)
# 方法二：序贯模型
# 注：在这里不太适合这种方法，因为后面会用到每一层的名字，这里没法写。
# from keras.models import Sequential
# from keras.layers import Activation
# model = Sequential([
#     Dense(128,input_shape=(input)),
#     Activation('relu'),
#     Dense(64),
#     Activation('relu'),
#     Dense(128),
#     Activation('relu'),
#     Dense(784),
#     Activation('softmax'),
# ])

# 方法三：class方法
# 注：同样不适用
# import keras
# from keras.layers import Input
# from keras import Model
# input = Input(shape = (784,))
# class mymodel(keras.Model):
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.hidden1 = keras.layers.Dense(128,activation='relu')
#         self.hidden2 = keras.layers.Dense(64, activation='relu')
#         self.hidden3 = keras.layers.Dense(128, activation='relu')
#         self.output = keras.layers.Dense(784,activation='softmax')
#
#     def call(self,inputs):
#         hidden1 =self.hidden1(input)
#         hidden2 = self.hidden2(hidden1)
#         hidden3 = self.hidden3(hidden2)
#         output = self.output(hidden3)
#
#         return output
# model= mymodel()


model.compile(
    optimizer= 'adam',
    loss= 'mse'
)
history = model.fit(x_train, x_train,
                    batch_size=128,
                    epochs=5,
                    verbose=2,
                    validation_data = (x_test,x_test)
)
#----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------保存模型、模型可视化-------------------------------------------------------#
from keras.utils import plot_model
# 保存模型
model.save('model_MultipleLayerAutoEnconder.h5')
# 模型可视化
plot_model(model, to_file='model_MultipleLayerAutoEnconder.png', show_shapes=True)
#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------查看自编码器的编码效果---------------------------------------------------#
# encoder做测试
conv_encoder = Model(input,hidden2)
encoder_imgs = conv_encoder.predict(x_test)
# 打印10张测试集手写数字的压缩效果
n = 10
import matplotlib.pyplot as plt

plt.figure(figsize = (20,8))
for i in range(n):
    ax = plt.subplot(1,n,i+1)
    plt.imshow(encoder_imgs[i].reshape(4,16).T) # 8*8 的特征，转化为 4*16的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#----------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------查看自编码器的解码效果---------------------------------------------------#
# decoder做测试
decoder_imgs = model.predict(x_test)
# 打印10张测试集手写数字的压缩效果
n = 10
plt.figure(figsize = (20,6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test[i].reshape(28,28).T)        # 将原图转化为 28*28的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # 打印解码图
    x = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoder_imgs[i].reshape(28, 28).T)  # 将解码图转化为 28*28的图像
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