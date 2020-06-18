# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月17日
# 内容:LSTM实现数据的分类
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建正则化自编码器模型
# 4、训练模型
# 5、模型可视化
# 6、查看自编码器的解码效果
# 7、训练过程可视化
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense,LSTM
from keras.utils import to_categorical
from keras.models import Sequential
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#minist的image是28*28的维度，这里定义LSTM的输入维度为（28，），
# 将image一行一行的输入到LSTM的cell中，这样time_step=28,表示一个image有28行，LSTM的输出维度是30
nb_lstm_outputs=30
nb_time_steps=28
nb_input_cector=28
path = 'D:\\northwest\\小组视频\\5单层自编码器\\mnist.npz'
f = np.load(path)
X_train = f['x_train']
X_test = f['x_test']
Y_train = f['y_train']
Y_test = f['y_test']
X_train = X_train.astype('float32')/255#数据预处理，归一化
X_test = X_test.astype('float32')/255
Y_train = to_categorical(Y_train,num_classes=10)
Y_test = to_categorical(Y_test,num_classes=10)
#  ---------------------  2、读取手写体数据及与图像预处理 -------------------
#  ---------------------  3、构建自编码器模型 ------------------------------
model = Sequential()
model.add(LSTM(units=nb_lstm_outputs,input_shape=(nb_time_steps,nb_input_cector)))
model.add(Dense(10,activation='softmax'))
#  ---------------------  3、构建自编码器模型 ------------------------------
#  ---------------------- 4、模型训练 ----------------------------------------
model.compile(loss='categorical_crossentropy',optimizer='adam')
history=model.fit(X_train,Y_train,
                  epochs=3,batch_size=128,verbose=1,
                   validation_data = (X_test, Y_test))
model.summary()
#score=model.evaluate(X_test,Y_test,batch_size=128,verbose=1)
#print(score)
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()
#  ---------------------- 4、模型训练 ----------------------------------------