#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.18
#software：pycharm
#项目名称：LSTM（keras）
#--------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras等包
# 2、数据集导入
# 3、模型建立
# 4、模型训练
# 5、评价
# 6、模型保存和预测
# ----------------------   代码布局： ----------------------



#  -------------------------- 1、导入需要包 -------------------------------

from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.models import Sequential
import matplotlib.pyplot as plt

#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------
#parameters for LSTM
nb_lstm_outputs = 30  # 神经元个数
nb_time_steps = 28    # 时间序列长度
nb_input_vector = 28  # 输入序列

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  -------------------------- 3、构建lstm模型 ------------------------------

#build model
model = Sequential()
model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#  -------------------------- 3、构建lstm模型 ------------------------------

#  -------------------------- 4、训练 ------------------------------
#train: epcoch, batch_size
model.fit(x_train, y_train, epochs=20, batch_size=128, verbose=1)

#  -------------------------- 5、评价 ------------------------------

model.summary()    # 可以使用model.summary()来查看神经网络的架构和参数量等信息
score = model.evaluate(x_test, y_test,batch_size=128, verbose=1)
print(score)
print(score.history.keys())
plt.plot(score.history['loss'])
plt.plot(score.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  -------------------------- 5. 评价  ------------------------------