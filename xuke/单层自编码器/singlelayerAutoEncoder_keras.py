#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.5
#software：pycharm
#项目名称：单层自编码器（keras）
#--------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包  sss panda是一个可数据预处理的包
# 2、手写数据集导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape

#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------
path = 'D:\\keras\\编码器\\mnist.npz'           # 数据集路径
f = np.load(path)                              # 打开文件  以npz结尾的数据集是压缩文件，里面还有其他的文件
# 取出60000个训练集，10000个测试集
X_train=f['x_train']        # 训练数据
X_test=f['x_test']          # 测试数据
f.close()                   # 关闭文件

# 输出观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
# 输出
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 数据准备
# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))     #numpy中reshape函数的三种常见相关用法：reshape(1,-1)转化成1行：reshape(2,-1)转换成两行：reshape(-1,1)转换成1列：reshape(-1,2)转化成两列
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、构建单层自编码器模型 ---------------------
#  --------------------- 3.1、Sequential() ---------------------

model = Sequential()
model.add(Dense(units = 64,activation='relu',input_shape=(input,)))
model.add(Dense(units = 784,activation='sigmode'))
model.compile(optimizer='adam',metrics=['acc'])  # 编译模型

#  --------------------- 3.1、Sequential() ---------------------

#  ------------------------- 3.2、API --------------------------

inputs=Input(shape=(input,))
hidden=Dense(64,activation='relu')         # 784-64
output=Dense(784,activation='sigmode')     # 64-784输出，sigmode激活
model = Model(inputs=input, outputs=output)
model.compile(optimizer='adam',metrics=['acc'])  # 编译模型

#  ------------------------- 3.2、API --------------------------

#  ------------------------- 3.3、class类继承--------------------------

class singleautoencoder(keras.Model):
    def __init__(self):
        super(singleautoencoder, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu')      # 隐藏层
        self.dense2 = keras.layers.Dense(784, activation='sigmode')  # 输出层

    def call(self, inputs, mask=None):
        x = self.dense1(input)
        x = self.dense2(x)
        return(x)
model= singleautoencoder()

#  ------------------------- 3.3、class类继承--------------------------
#  --------------------- 3、构建单层自编码器模型 ---------------------


#  --------------------- 4、模型可视化 ---------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))   # 可视化模型

#  --------------------- 4、模型可视化 ---------------------

#  --------------------- 5、训练 ---------------------
epochs = 5
batch_size = 128
history = model.fit(X_train, X_train,batch_size=batch_size,epochs=epochs, verbose=1,validation_data=(X_test, X_test))  # 训练模型
#  --------------------- 5、训练 ---------------------

#  --------------------- 6、查看自编码器的压缩效果 ---------------------
import matplotlib.pyplot as plt
# 为隐藏层的结果
conv_encoder = Model(input, hidden)  # 只取编码器做模型 (取输入层x和隐藏层h，作为网络结构)
encoded_imgs = conv_encoder.predict(X_test)

# 打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 16).T)  # 8*8 的特征，转化为 4*16的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#  --------------------- 6、查看自编码器的压缩效果 ---------------------

#  --------------------- 7、查看自编码器的解码效果 ---------------------
# decoded_imgs 为输出层的结果
decoded_imgs = model.predict(X_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 打印解码图
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28)) # 784 转换为 28*28大小的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#  --------------------- 7、查看自编码器的解码效果 ---------------------

#  --------------------- 8、训练过程可视化 ---------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------- 8、训练过程可视化 ---------------------


