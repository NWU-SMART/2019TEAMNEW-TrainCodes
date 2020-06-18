# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/17
#  @Author: MiJizong
#  @Content: 使用LSTM在手写体数据集上训练——Keras
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建LSTM模型
# 4、模型可视化
# 5、训练
# 6、训练过程可视化
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers.core import Dense
from keras.utils import np_utils
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第1块显卡
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'  # 允许副本存在
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\Office_software\\PyCharm\\datasets\\mnist.npz(本地路径)
path = 'D:\\Office_software\\PyCharm\\datasets\\mnist.npz'
f = np.load(path)
#  以npz结尾的数据集是压缩文件，里面还有其他的文件
#  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
x_train = f['x_train']
y_train = f['y_train']
# 测试数据
x_test = f['x_test']
y_test = f['y_test']
f.close()
# 数据放到本地路径test

# 数据预处理
#  归一化
x_train = x_train.reshape(-1, 28, 28) / 255  # -1代表未知
x_test = x_test.reshape(-1, 28, 28) / 255

# 将标签转换为分类的 one-hot 编码
y_train = np_utils.to_categorical(y_train,num_classes=10)  # y为int数组，num_classes为标签类别数
y_test = np_utils.to_categorical(y_test,num_classes=10)

# 参数声明
input_height = 28
input_width = 28
output_size = 10
hidden_layer = 64
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3.1、构建LSTM Sequential模型 ----------------------
lstm1 = Sequential()
lstm1.add(LSTM(batch_input_shape=(None,input_height,input_width),
               output_dim=hidden_layer,
               activation='relu'))
lstm1.add(Dense(output_size, activation='softmax'))

#  --------------------- 3.1、构建LSTM Sequential模型 ----------------------

#  -------------------------- 3.2、构建LSTM API模型 ------------------------

# 定义神经网络层数
x = Input(shape=(input_height, input_width))
h = LSTM(output_dim=hidden_layer,activation='relu')(x)
r = Dense(output_size,activation='softmax')(h)

# 构建模型，给定模型优化参数
lstm2 = Model(inputs=x,outputs=r)

#  -------------------------- 3.2、构建LSTM模型 API模型 --------------------

#  --------------------- 3.3、构建LSTM模型 class继承模型 --------------------
# 定义神经网络层数
inputs = Input(shape=(input_height, input_width))

class Lstm(keras.Model):
    def __init__(self):          # 初始化
        super(Lstm, self).__init__()
        self.lstm = LSTM(output_dim=hidden_layer,activation='relu')
        self.dense = Dense(output_size,activation='softmax')

    def call(self, inputs):  # 实例化调用
        t = self.lstm(inputs)
        return self.dense(t)

lstm3 = Lstm()  # 实例化
print(lstm3)

#  --------------------- 3.3、构建LSTM模型 class继承模型 --------------------

#  ---------------------------- 4、模型可视化 ------------------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# 可视化模型
SVG(model_to_dot(lstm1).create(prog='dot', format='svg'))

#  ---------------------------- 4、模型可视化 ------------------------------

#  ----------------------------- 5、训练 -----------------------------------

# 设定peochs和batch_size大小
epochs = 5
batch_size = 128

lstm1.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

# 训练模型
history = lstm1.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs, verbose=1,
                          validation_data=(x_test, y_test)
                         )

#  ----------------------------- 5、训练 -----------------------------------

#  --------------------------- 6、训练过程可视化 ----------------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------------- 6、训练过程可视化 ----------------------------


