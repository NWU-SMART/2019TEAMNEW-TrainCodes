# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月12日
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、训练模型
# 5、查看自编码器的解码效果
# 6、训练过程可视化
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
from IPython.core.display import SVG
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import Model
import numpy as np
from keras import Sequential
from keras.utils import plot_model, model_to_dot
import matplotlib.pyplot as plt
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
path = '...'
f = np.load(path)
X_train = f['x_train']
X_test = f['x_test']
f.close()
print(X_train.shape)
print(X_test.shape)
#数据格式进行转换
x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')/255#数据预处理，归一化
X_test = X_test.astype('float32')/255
print('X_train shape:',X_train.shape)
print(X_train.shape[0],'train samples')
print(X_test.shape[0],'test samples')
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建卷积自编码器模型 ---------------------------
model = Sequential()
#编码
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))#通道变为16，即28*28*1--28*28*16
model.add(MaxPooling2D((2,2),padding='same'))#28*28*16--14*14*16
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))#14*14*16---14*14*8
model.add(MaxPooling2D((2,2),padding='same'))#14*14*8--7*7*8
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))#7*7*8--7*7*8
model.add(MaxPooling2D((2,2),padding='same'))#7*7*8--4*4*8
#解码
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))#4*4*8--4*4*8
model.add(UpSampling2D((2,2)))#4*4*8--8*8*8
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))#8*8*8--8*8*8
model.add(UpSampling2D((2,2)))#8*8*8--16*16*8
model.add(Conv2D(16, (3, 3), activation='relu'))#16*16*8--14*14*16,每加same
model.add(UpSampling2D((2,2)))#14*14*16--28*28*16
model.add(Conv2D(1, (3, 3), activation='sigmoid',padding='same'))
#Model = model()
model.compile(optimizer='adadelta', loss='binary_crossentropy')
#  ---------------------  3、构建卷积自编码器模型 ---------------------------
#  ---------------------  4、模型训练----------- ---------------------------
SVG(model_to_dot(model).create(prog='dot', format='svg'))
epochs = 3
batch_size = 128
history = model.fit(X_train, X_train, batch_size=batch_size,
                      epochs = epochs,
                      verbose = 1,
                      validation_data=(X_test, X_test))
#  ---------------------  4、模型训练----------- ---------------------------
#  -----------------------5、查看解码效果 ----------------------------------
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
#  -----------------------5、查看解码效果 ----------------------------------
#  -----------------------6、训练过程可视化 --------------------------------
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#  -----------------------6、训练过程可视化 --------------------------------