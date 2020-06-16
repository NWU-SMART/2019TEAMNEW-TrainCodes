#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.16
#software：pycharm
#项目名称：去噪自编码器（keras）
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
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras import Model
#  -------------------------- 1、导入需要包 -------------------------------
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'D:\\keras_datasets\\mnist.npz'
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

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

# 加入噪声数据

noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、构建去噪自编码器模型 ---------------------
input = Input(shape = (28,28,1))
# 编码
conv1 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(input)  # 1*28*28 --> 32*28*28
pool1 = MaxPooling2D(pool_size = (2,2),padding='same')(conv1)         # 32*28*28 -->32*14*14
conv2 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(pool1)   # 32*14*14 --> 32*14*14
pool2 = MaxPooling2D(pool_size = (2,2),padding='same')(conv2)         # 32*14*14  --> 32*7*7
# 解码
conv3 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(pool2)   # 32*7*7 --> 32*7*7
up1 = UpSampling2D((2,2))(conv3)                                      # 32*7*7-->32*14*14
conv4 = Conv2D(32,(3, 3), padding='same',activation = 'relu')(up1)     # 32*14*14 --> 32*14*14
up2 = UpSampling2D((2,2))(conv4)                                      # 32*14*14 --> 32*28*28

output = Conv2D(1,(3, 3),padding='same',activation = 'sigmoid')(up2)   # 32*28*28--> 1*28*28
model = Model(inputs=input, outputs=output)
model.compile(optimizer= 'adadelta',loss= 'binary_crossentropy')

#  --------------------- 3、构建去噪自编码器模型 ---------------------

#  --------------------- 4、训练 ---------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))
epochs = 3
batch_size = 128
history = model.fit(X_train_noisy, X_train,batch_size=batch_size,epochs=epochs, verbose=1,validation_data=(X_test_noisy, X_test))

#  --------------------- 4、训练 ---------------------


#  --------------------- 5、查看解码效果 ---------------------

# decoded_imgs 为输出层的结果
decoded_imgs = model.predict(X_test_noisy)

n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
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

#  --------------------- 5、查看解码效果 ---------------------


#  --------------------- 6、训练过程可视化 --------------------

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------- 6、训练过程可视化 ---------------------