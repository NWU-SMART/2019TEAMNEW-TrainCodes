# ----------------开发者信息---------------------------------------------------
# 开发者：张春霞
# 开发日期：2020年5月26日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息------------------------------------------------------
#  -------------------------- 1、导入需要包 --------------------------------------
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers, Input, Model  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from numexpr import evaluate
from prompt_toolkit import history
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#  -------------------------- 1、导入需要包 ------------------------------------------

#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
# 数据存放路径

path = 'D:/northwest/小组视频/1房屋预测/boston_housing.npz'
#加载数据
f = np.load(path)
#训练数据
x_train=f['x'][:404]#训练集取0-404
y_train=f['y'][:404]#训练集取0-404
#测试数据
x_test=f['x'][404:]#验证集取404-505
y_test=f['y'][404:]#验证集取404-505
f.close()
#转成DATAframe方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_test_pd = pd.DataFrame(x_test)
y_test_pd = pd.DataFrame(y_test)
print(x_train_pd.head(5))
print('------------')
print(y_train_pd.head(5))
#  -------------------------- 2、房价训练和测试数据载入 -------------------------------

#  -------------------------- 3、数据归一化 ------------------------------------------
#训练数据归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)
#验证数据归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_test_pd)
x_test = min_max_scaler.transform(x_test_pd)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)
#  -------------------------- 3、数据归一化 ------------------------------------------

#  -------------------------- 4、模型训练   ------------------------------------------
#-------------------------------API方法------------------------------------------------
inputs = Input(shape=(404,))
#层的实例是可以调用的，它以张量为参量，并且返回一个张量
x = Dense(units=15,activation='relu')(inputs)
x = Dense(units=15,activation='relu')(x)
predictions = Dense(units=1,activation='linear')(x)
#创建一个包含输入层和三个全连接层的模型
model = Model(input=inputs,output=predictions)
model.compile(loss='MSE', optimizer='adam')
model.fit(x_train, y_train, epochs=200, batch_size=200, verbose=2, validation_data=(x_test, y_test))#开始训练
#-------------------------------API方法------------------------------------------------
#-----------------------------sequential方法-----------------------------------------
model1 = Sequential()
model1.add(Dense(15,activation='relu',input_shape=(x_train_pd.shape[1])))
model1.add(Dropout(0.5))
model1.add(Dense(20,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(units=1, activation='linear'))
model1.compile(loss='MSE', optimizer='adam')
model1.fit(x_train,y_train,epochs=200,batch_size=200)
score = model1(evaluate(x_test,y_test,batch_size=200 ))
#-----------------------------sequential方法-----------------------------------------
#-----------------------------class方法---------------------------------------------
class housepredict(keras.model2):
     def __init__(self, use_dp=True):
        super(housepredict,self).__init__()
        self.dense1 = keras.layers.Dense(15,activation='relu')
        self.dropout = keras.layers.Dropout(0.5)
        self.dense2 = keras.layers.Dense(20,activation='relu')
        self.dense3 = keras.layers.Dense(1,activation='linear')

     def call(self,inputs):
         x = self.dense(inputs)
         x = self.dense2(x)
         x = self.dropout(x)
         x = self.dense3(x)
           #return(x)

model2 = housepredict()
model2.compile()
model2.fit(x_train, y_train, epochs=200, batch_size=200, verbose=2, validation_data=(x_test, y_test))
#-----------------------------class方法---------------------------------------------
#  -------------------------- 4、模型训练   ------------------------------------------

#  -------------------------- 5、模型可视化    --------------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#  -------------------------- 5、模型可视化    -------------------------------------

#  -------------------------- 6、模型保存和预测    ---------------------------------
from keras.utils import plot_model
from keras.models import load_model
model.save('model_MLP.h5')
plot_model(model,to_file='model_MLP.h5',show_shape=True)
model = load_model('model_MLP.h5')
y_new = model.predict(x_test)
min_max_scaler.fit(y_test_pd) #验证标签反归一化
y_new = min_max_scaler.inverse_transform(y_new)
#  -------------------------- 6、模型保存和预测    ---------------------------------