# -------------------------------------------------开发者信息----------------------------------------------------------#
# 开发者：高强
# 开发日期：2020年6月3日
# 开发框架：keras

#----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------代码布局-----------------------------------------------------------#
# 1、读取手写体数据集、图像预处理、加入噪声数据
# 2、构建自编码器模型
# 3、训练
# 4、保存模型及模型可视化
# 5、查看自编码器的压缩效果
# 6、查看自编码器的解码效果
# 7、训练过程可视化
#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------读取手写体数据及与图像预处理------------------------------------------------#
import numpy as np
# 载入数据：本地
path = 'F:\\Keras代码学习\\keras\\keras_datasets\\mnist.npz'
# 载入数据：服务器
# path = 'mnist.npz'
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
#----------------------------------------------构建噪声数据集----------------------------------------------------------#
'''
关于：np.random.normal
loc：float  此概率分布的均值（对应着整个分布的中心centre）
scale：float 此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints   输出的shape，默认为None，只输出一个值
'''
noise_factor = 0.5
x_train_noise = x_train + noise_factor * np.random.normal(loc = 0.0,scale = 1.0,size = x_train.shape)
x_test_noise = x_test + noise_factor * np.random.normal(loc = 0.0,scale = 1.0,size = x_test.shape)

'''
np.clip()的三个参数:
第一个为数组,
使用第二个参数代替数组中的最小数(比0小的全部替代为0)
使用第三个参数代替数组中的最大数(比1大的全部替代为1)
'''
x_train_noise = np.clip(x_train_noise,0.,1.) # 控制训练集的数据在0-1之间
x_test_noise = np.clip(x_test_noise,0.,1.)   # 控制测试集的数据在0-1之间


#----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------构建去噪自编码器模型------------------------------------------------------#


from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import Model
input = Input(shape = (28,28,1))
# encoder
conv1 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(input)  # 1*28*28 --> 32*28*28
pool1 = MaxPooling2D(pool_size = (2,2),padding='same')(conv1)         # 32*28*28 -->32*14*14
conv2 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(pool1)   # 32*14*14 --> 32*14*14
pool2 = MaxPooling2D(pool_size = (2,2),padding='same')(conv2)         # 32*14*14  --> 32*7*7

# decoder
conv3 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(pool2)   # 32*7*7 --> 32*7*7
up1 = UpSampling2D((2,2))(conv3)                                      # 32*7*7-->32*14*14
conv4 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(up1)     # 32*14*14 --> 32*14*14
up2 = UpSampling2D((2,2))(conv4)                                      # 32*14*14 --> 32*28*28

output = Conv2D(1,(3, 3),padding='same',activation = 'sigmoid')(up2)   # 32*28*28--> 1*28*28

model = Model(inputs=input, outputs=output)

model.compile(
    optimizer= 'adadelta',
    loss= 'binary_crossentropy'
)
history = model.fit(x_train_noise, x_train_noise,
                    batch_size=128,
                    epochs=3,
                    verbose=2,
                    validation_data=(x_test_noise,x_test_noise)
)
#----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------保存模型、模型可视化-------------------------------------------------------#
from keras.utils import plot_model
# 保存模型
model.save('keras_model_DenoiseAutoEnconder.h5')
# 模型可视化
plot_model(model, to_file='keras_model_DenoiseAutoEnconder.png', show_shapes=True)
#----------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------查看自编码器的解码效果---------------------------------------------------#
# decoder做测试
decoder_imgs = model.predict(x_test_noise)
# 打印10张测试集手写数字的压缩效果
import matplotlib.pyplot as plt
n = 10
plt.figure(figsize = (20,6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3,n,i+1)
    plt.imshow(x_test_noise[i].reshape(28,28))        # 将原图转化为 28*28的图像
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