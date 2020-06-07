#------------------------------------------------开发者信息--------------------------------------------
#开发人：王园园
#开发日期：2020.6.1
#开发软件：pycharm
#开发项目：去噪自编码器（keras）

#-----------------------------------------------代码布局-----------------------------------------------
#1、导包
#2、读取手写体数据及与图像预处理
#3、构建自编码器模型
#4、模型可视化
#5、训练
#6、查看编码效果
#7、训练过程可视化

#----------------------------------------------------导包----------------------------------------------
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, UpSampling2D
from keras.utils import model_to_dot
from networkx.drawing.tests.test_pylab import plt
from IPython.display import SVG
#----------------------------------------------------读取手写体数据及与图像预处理-------------------------


path = 'D:\\keras_datasets\\mnist.npz'
f = np.load(path)
x_train = f['x_train']
x_test = f['x_test']
f.close()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
#加入噪声数据
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

#----------------------------------------------------构建去噪自编码器模型--------------------------------
#编码器
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
#解码器
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(2, 2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(2, 2))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
model.compile(optimizer='adadelta', loss='binary_crossentropy')

SVG(model_to_dot(model).create(prog='dot', format='svg'))

#--------------------------------------------------------训练--------------------------------------------
epochs= 3
batch_size = 128
history = model.fit(x_train_noisy, x_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_data=(x_test_noisy, x_test))

#-------------------------------------------------------查看解码效果----------------------------------------
decoded_imgs = model.predict(x_test_noisy)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    #原图
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #解码效果图
    ax = plt.subplot(3, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#----------------------------------------------------------训练过程可视化-------------------------------------
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
plt.show()
