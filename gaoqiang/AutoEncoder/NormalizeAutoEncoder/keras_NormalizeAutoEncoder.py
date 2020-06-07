# -------------------------------------------------开发者信息----------------------------------------------------------#
# 开发者：高强
# 开发日期：2020年6月2日
# 开发框架：keras
# 温馨提示：
#----------------------------------------------------------------------------------------------------------------------#
# --------------------------------------------------代码布局-----------------------------------------------------------#
# 1、读取手写体数据及与图像预处理
# 2、构建自编码器模型
# 3、训练
# 4、保存模型及模型可视化
# 5、查看自编码器的解码效果
# 6、训练过程可视化
#----------------------------------------------------------------------------------------------------------------------#
'''
正则化说明：就整体而言，对比加入正则化和未加入正则化的模型，训练输出的loss和Accuracy信息，可以发现，加入正则化后，loss下降
的速度会变慢，准确率Accuracy的上升速度会变慢，并且未加入正则化模型的loss和Accuracy的浮动比较大（或者方差比较大），而加入正
则化的模型训练loss和Accuracy，表现的比较平滑。并且随着正则化的权重lambda越大，表现的更加平滑。这其实就是正则化的对模型的
惩罚作用，通过正则化可以使得模型表现的更加平滑，即通过正则化可以有效解决模型过拟合的问题。

'''

#----------------------------------------读取手写体数据及与图像预处理--------------------------------------------------#
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
# np.prod是将28*28的矩阵转换为1*784，方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))  # 60000*784
x_test  = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))     # 10000*784

#----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------构建正则自编码器模型------------------------------------------------------#

from keras.layers import Input
from keras.layers import Dense
from keras import Model
from keras import regularizers
input = Input(shape = (784,))
hidden = Dense(32,activation = 'relu',activity_regularizer= regularizers.l1(10e-5))(input)  # l1正则
output = Dense(784,activation = 'sigmoid')(hidden)     # decoder

model = Model(inputs=input, outputs=output)



model.compile(
    optimizer= 'adam',
    loss= 'mse'
)
history = model.fit(x_train, x_train,
                    batch_size=128,
                    epochs=15,
                    verbose=2,
                    validation_data = (x_test,x_test)
)
#----------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------保存模型、模型可视化-------------------------------------------------------#
from keras.utils import plot_model
# 保存模型
model.save('keras_model_NormalizeAutoEnconder.h5')
# 模型可视化
plot_model(model, to_file='keras_model_NormalizeAutoEnconder.png', show_shapes=True)
#----------------------------------------------------------------------------------------------------------------------#

#---------------------------------------------查看自编码器的解码效果---------------------------------------------------#
# decoder做测试
import matplotlib.pyplot as plt
decoder_imgs = model.predict(x_test)
# 打印10张测试集手写数字的压缩效果
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



