#-------------------------------------------------------开发者信息----------------------------------------
#开发人：王园园
#开发日期：2020.5.30
#开发软件：pycharm
#开发项目：正则自编码器（keras）

#----------------------------------------------------代码布局--------------------------------------------
#1、导包
#2、读取手写体数据及与图像预处理
#3、构建自编码器模型
#4、模型可视化
#5、训练
#6、查看解码效果
#7、训练过程可视化

#----------------------------------------------------导包------------------------------------------------
import numpy as np
from keras import Input, Sequential, regularizers
from keras.layers import Dense
from networkx.drawing.tests.test_pylab import plt

#-----------------------------------------------------读取手写体数据及与图像预处理---------------------------
path = 'D:/keras_datasets/mnist.npz'
f = np.load(path)
x_train = f['x_train']  #训练数据
x_test = f['x_test']    #测试数据
f.close()

#数据预处理，标准化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
#数据准备，将28*28矩阵转换成1*784， 方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

#----------------------------------------------------构建正则自编码器模型------------------------------------
input_size = 784
hidden_size = 32
output_size = 784

model = Sequential()
model.add(Dense(hidden_size, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
model.add(Dense(output_size, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse')

#------------------------------------------------------训练-------------------------------------------------
epochs = 15
batch_size = 128
history = model.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, x_test))

#-----------------------------------------------------查看解码效果--------------------------------------------
decoded_imgs = model.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # 解码效果图
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
#----------------------------------------------------训练过程可视化---------------------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()