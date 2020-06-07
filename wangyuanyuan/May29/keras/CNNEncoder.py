#----------------------------------------------------开发者信息----------------------------------
#开发者：王园园
#开发日期：2020.5.29
#开发软件：pycharm
#项目：卷积自编码器（keras）

#---------------------------------------------------代码布局---------------------------------
#1、导入包
#2、读取手写体数据及与图像预处理
#3、构建自编码器模型
#4、模型可视化
#5、训练
#6、产看解码效果
#7、训练过程可视化

#-----------------------------------------------------导包-------------------------------
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.utils import model_to_dot
from networkx.drawing.tests.test_pylab import plt
#----------------------------------------------------读取手写体数据及与图像预处理-------------------


path = 'D:/keras_datasets/mnist.npz'
f = np.load(path)
#60000个训练，10000个测试
x_train = f['x_train']
x_test = f['x_test']
f.close()

#数据格式进行转换
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.reshape[0], 28, 28, 1)

#数据预处理，标准化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#---------------------------------------------------构建卷积自编码器---------------------------------
modelEncoder = Sequential()
#编码
modelEncoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
modelEncoder.add(MaxPooling2D((2, 2), padding='same'))
modelEncoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
modelEncoder.add(MaxPooling2D((2, 2), padding='same'))
modelEncoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
modelEncoder.add(MaxPooling2D((2, 2), padding='same'))

#解码
modelEncoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
modelEncoder.add(UpSampling2D((2, 2)))
modelEncoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
modelEncoder.add(UpSampling2D(2, 2))
modelEncoder.add(Conv2D(16, (3, 3), activation='relu'))
modelEncoder.add(UpSampling2D((2, 2)))
modelEncoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

modelDe = modelEncoder()
modelDe.compile(optimizer='adadelta', loss='binary_crossentropy')

#--------------------------------------------------模型可视化及训练------------------------------------------
SVG(model_to_dot(modelDe).create(prog='dot', format='svg'))
epochs = 3
batch_size = 128
history = modelDe.fit(x_train, x_train, batch_size=batch_size,
                      epochs = epochs,
                      verbose = 1,
                      validation_data=(x_test, x_test))

#--------------------------------------------------查看解码效果----------------------------------------------
decoded_imgs = modelDe.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #解码效果图
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#-----------------------------------------------训练过程可视化----------------------------------------------
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()







