#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.15
#software：pycharm
#项目名称：正则自编码器（keras）
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
import keras
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.layers import Dense
from keras import Model
from keras import regularizers
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'D:\\keras\\编码器\\mnist.npz'
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

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))


#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、构建正则自编码器模型 ---------------------
input_size = 784
hidden_size = 32
output_size = 784
#  --------------------- 3.1 sequential() ---------------------
model = Sequential()
model.add(Dense(32, activation='relu', activity_regularizer=regularizers.l1(10e-5)))
model.add(Dense(784, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse')
#  --------------------- 3.1 sequential() ---------------------

#  ---------------------------- 3.2 API -----------------------
input = Input(shape = (784,))
hidden = Dense(32,activation = 'relu',activity_regularizer= regularizers.l1(10e-5))(input)
output = Dense(784,activation = 'sigmoid')(hidden)
model = Model(inputs=input, outputs=output)
model.compile(optimizer= 'adam',loss= 'mse')
#  ------------------------ 3.2 API ----------------------------

#  --------------------- 3.3、构建正则自编码器class继承模型 -----------------
inputs = Input(shape=(input_size,))
class NormalAutoEncoder(keras.Model):
    def __init__(self):
        super(NormalAutoEncoder,self).__init__()
        self.dense1 = Dense(32,activation='relu',activity_regularizer=regularizers.l1(10e-5))    # L1正则
        self.dense2 = Dense(784,activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
model = NormalAutoEncoder()
model.compile(optimizer='adam',loss='mse')

#  --------------------- 3.3、构建正则自编码器class继承模型 -----------------

#  --------------------- 3、构建正则自编码器模型 ---------------------

#  ----------------------- 4、训练 ----------------------------------

epochs = 15
batch_size = 128
history = model.fit(X_train, X_train,batch_size=batch_size, epochs=epochs,verbose=1,validation_data=(X_test, X_test))
# 梯度下降  迭代次数  日志显示  验证集
#  ------------------------------ 4、训练 ---------------------------

#  --------------------- 5、查看解码效果 ---------------------

# decoded_imgs 为输出层的结果
decoded_imgs = model.predict(X_test)
n = 10
plt.figure(figsize=(20, 6))   # 图像的宽和高
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i+1)   # 创建成1*n的网格
    plt.imshow(X_test[i].reshape(28, 28))   # 转化成28*28
    plt.gray()    # 灰度显示
    ax.get_xaxis().set_visible(False)   # 不显示x轴
    ax.get_yaxis().set_visible(False)   # 不显示y轴
    # 解码效果图
    ax = plt.subplot(3, n, i+n+1)
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