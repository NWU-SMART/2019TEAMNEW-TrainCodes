#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月15日
# 开发框架：keras
# 开发内容：使用LSTM网络实现手写数字识别(三种方法）
#----------------------------------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, os 的包
# 2、读取数据和数据处理
# 3、参数定义
# 4、built the LSTM model
# 5、模型训练
# 6、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, LSTM
import keras
from keras import Input
from keras.models import Model
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
#  -------------------------- 导入需要包 -------------------------------

#  -------------------------- 2、读取数据和数据处理-------------------------------
# 数据集本地路径
path = 'E:\\study\\kedata\\mnist.npz'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
x_train = f['x_train']
y_train = f['y_train']
# 测试数据
x_test = f['x_test']
y_test = f['y_test']
f.close()

# 数据reshape和归一化
x_train = x_train.reshape(-1, 28, 28) / 255
x_test = x_test.reshape(-1, 28, 28) / 255
# 将类型信息进行one-hot编码(10类)
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
#  -------------------------- 读取数据和数据处理-------------------------------

#  -------------------------- 3、参数定义-------------------------------
TIME_STEPS = 28  # as same as the image height
INPUT_SIZE = 28  # as same as the image width
BATCH_SIZE = 128
OUTPUT_SIZE = 10
CELL_SIZE = 50  # how many hidden layer
LR = 0.001
EPOCHS = 5
#  -------------------------- 参数定义-------------------------------

#  -------------------------- 4、built the LSTM model-------------------------------
''' Sequentual方法：
model = Sequential()
model.add(LSTM(batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
                    output_dim=CELL_SIZE,
                    activation='relu'))
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))
'''
''' Class方法：
input = Input(shape=(None, TIME_STEPS, INPUT_SIZE))
class Lstm(keras.Model):
    def __init__(self):
        super(Lstm, self).__init__(name='Lstm')
        self.lstm = keras.layers.LSTM(output_dim=CELL_SIZE)
        self.relu = keras.layers.ReLU()
        self.dense = keras.layers.Dense(OUTPUT_SIZE)
        self.softmax = keras.layers.Softmax()

    def call(self, input):
        x = self.lstm(input)
        x = self.relu(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x
model = Lstm()
'''
# API方法：
# input = Input(shape=(None, TIME_STEPS, INPUT_SIZE))
input = Input(shape=(TIME_STEPS, INPUT_SIZE))
# 这里要去掉None,否则会报错：ValueError: Input 0 is incompatible with layer lstm_1: expected ndim=3, found ndim=4

x = LSTM(output_dim=CELL_SIZE, activation='relu')(input)
x = Dense(OUTPUT_SIZE, activation='softmax')(x)
model = Model(inputs=input, outputs=x)

# 优化器，损失
adam = Adam(LR)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

#  -------------------------- built the LSTM model-------------------------------

#  -------------------------- 5、模型训练-------------------------------
# training
history = model.fit(x_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS, verbose=1,
                    validation_data=(x_test, y_test)
                    )
#  -------------------------- 模型训练-------------------------------

#  --------------------- 6、训练过程可视化 ---------------------

print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  ---------------------训练过程可视化 ---------------------
