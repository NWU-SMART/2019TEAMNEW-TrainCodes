#--------------         开发者信息--------------------------
#开发者：王园园
#开发日期：2020.5.23
#software：pycharm
#项目名称：房价预测（keras）

#--------------------------导入包--------------------------
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers, Input, Model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#-------------------------加载数据--------------------------
from tensorflow import keras

path = 'D:\\keras_datasets\\boston_housing.npz'  #数据地址
f = np.load(path)
#404训练数据
x_train = f['x'][:404]   #训练数据0-404
y_train = f['y'][:404]   #训练标签0-404
x_valid = f['x'][404:]   #验证数据405-505
y_valid = f['y'][404:]   #验证标签
f.close()

#--------------------------数据处理---------------------------
#将数据转成DataFrame格式
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  #取测试数据的前5个
print(y_train_pd.head(5))  #取测试标签的前5个

#用MinMaxScaler()将数据归一化，归一化到[0,1]
#训练数据归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
#训练标签归一化
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)
#验证数据归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)
#验证标签归一化
min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

#------------------------Sequential()类型的模型--------------------------------------
model = Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(x_train_pd.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1, activation='linear'))
print(model.summary())                       #模型输出
model.compile(loss='MSE', optimizer='adam')  #模型编译
history = model.fit(x_train, y_train, epochs=200, batch_size=200, verbose=2, validation_data=(x_valid, y_valid))

#--------------------------API类型---------------------------------------------------
input1 = Input(shape=(404,))
X = Dense(units=10, activation='relu', input_shape=(x_train_pd.shape[1]))(input)
X = Dropout(0.2)(X)
X = Dense(units=15, activation='relu')(X)
X = Dense(units=1, activation='linear')(X)
model1 = Model(inputs=input, outputs=X)
model1.compile(loss='MSE', optimizer='adam')
model1.summary()
#verbose:日志显示，verbose = 0，在控制台没有任何输出；verbose = 1 ：显示进度条；verbose =2：为每个epoch输出一行记录
model1.fit(x_train, y_train, epochs=200, batch_size=200, verbose=2, validation_data=(x_valid, y_valid))

#----------------------------model(class)类继承-----------------------------------------------
input2 = Input(shape=(404,))
class HousePredict(keras.Model):
    def __init__(self, use_dp=True):
        super(HousePredict, self).__init__(name='MLP')
        self.use_dp = use_dp    #布尔变量，是否进行dropout

        self.dense1 = keras.layers.Dense(10, activation='relu')
        self.dense2 = keras.layers.Dense(15, activation='relu')
        self.dense3 = keras.layers.Dense(1, activation='linear')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.2)

    def call(self, input2):
        x = self.dense1(input2) #顺序
        if self.use_dp:
            x = self.dp(x)
        x = self.dense2(x)
        x = self.dense3(x)

model2 = HousePredict()   #实例化
model2.compile()          #编译
model.fit(x_train, y_train, epochs=200, batch_size=200, verbose=2, validation_data=(x_valid, y_valid))

#---------------------------------------模型可视化----------------------------------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])       #训练损失
plt.plot(history.history['val_loss'])   #验证损失
plt.title('Model loss')                 #标题
plt.ylabel('Loss')                      #纵坐标
plt.xlabel('Epoch')                     #横坐标
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

from keras.utils import plot_model
from keras.models import load_model
model.save('model_MLP.h5')                                      #保存模型
plot_model(model, to_file='model_MLP.png', show_shapes=True)    #模型可视化
model = load_model('model_MLP.h5')                              #加载模型
y_new = model.predict(x_valid)                                  #模型预测
min_max_scaler.fit(y_valid_pd)                                  #验证标签反归一化
y_new = min_max_scaler.inverse_transform(y_new)