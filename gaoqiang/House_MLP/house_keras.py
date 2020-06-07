# ----------------开发者信息--------------------------------#
# 开发者：高强
# 开发日期：2020年5月21日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------------代码布局---------------------------#
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测

# --------------------------导入包----------------------------#
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np

# -------------载入数据集----------#
path = 'F:\\Keras代码学习\\keras\\keras_datasets\\boston_housing.npz'  # 路径
f = np.load(path)  # 载入

# ---------------对数据进行预处理--------------------#
# 划分训练集和测试集（404个做训练，102个做测试，一共13个特征）
x_train = f['x'][:404]
y_train = f['y'][:404]
x_test = f['x'][404:]
y_test = f['y'][404:]
f.close()

# 转换成DataFrame数据
# （DataFrame是一个表格型的数据类型，每列值类型可以不同，是最常用的pandas对象。）
import pandas as pd

x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_test_pd = pd.DataFrame(x_test)
y_test_pd = pd.DataFrame(y_test)
# 在用Pandas读取数据之后，往往想要观察一下数据读取是否准确，这就要用到Pandas
# 里面的head( )函数，head( )函数默认只能读取前五行数据。
print(x_train_pd.head())
print('-------------------')
print(y_train_pd.head())

# -------------数据归一化----------#
# MinMaxScaler：归一到 [ 0，1 ] ；MaxAbsScaler：归一到 [ -1，1 ]
from sklearn.preprocessing import MinMaxScaler

# 训练集
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 测试集
min_max_scaler.fit(x_test_pd)
x_test = min_max_scaler.transform(x_test_pd)
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)

# -------------模型----------#
# --------------------方法一：序贯模型（从头到尾结构顺序，不分叉）-------------#
# model = Sequential([

#     Dense(10,input_shape=(x_train_pd.shape[1],)),
#     Activation('relu'),
#     Dropout(0.2),
#     Dense(15),
#     Activation('relu'),
#     Dense(1),
#     Activation('linear'), # 线性激励函数 回归一般在输出层用这个激励函数
# ])

# -----------或使用model.add这样的方法------------#
# model = Sequential()
# model.add(Dense(units=10,
#                 activation = 'relu', # activation要小写，不然报错
#                 input_shape=(x_train_pd.shape[1],)
#                )
#          )
# model.add(Dropout(0.2))
# model.add(Dense(units=15,
#                 activation ='relu'
#                )
#          )
# model.add(Dense(units=1,
#                 activation= 'linear'
#                )
#          )


# ---------------方 法二：Model式模型（使用函数式API的Model类模型）------------#

from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(x_train_pd.shape[1],))
x = Dense(10, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(15, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs=inputs, outputs=predictions)

# ---------------方 法三：Model类继承（class）------------#

# import keras
# from keras.layers import Input
# from keras.models import Model
# inputs = Input(shape=(x_train_pd.shape[1],))

# class mymodel(keras.Model):
#     def __init__(self):
#         super(mymodel,self).__init__()
#         self.dense1= keras.layers.Dense(10,activation='relu')
#         self.dropout=keras.layers.Dropout(0.2)
#         self.dense2= keras.layers.Dense(15,activation='relu')
#         self.dense3= keras.layers.Dense(1,activation='linear')

#     def call(self,inputs):
#         x=self.dense1(inputs)
#         x=self.dense2(x)
#         x=self.dropout(x)
#         x=self.dense3(x)

#         return x


# model= mymodel()
# ----------------------------------------------------------------------------#


model.compile(optimizer='adam',
              loss='mse',
              )
history = model.fit(x_train, y_train,
                    epochs=200,
                    batch_size=200,
                    verbose=2,  # 0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                    validation_data=(x_test, y_test)
                    )

print(model.summary())  # 打印网络结构

# ----------------模型可视化--------------------------#
import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# ----------------保存模型--------------------------#
from keras.utils import plot_model

# 保存模型
model.save('model_House_MLP.h5')
# 模型可视化
plot_model(model, to_file='model_House_MLP.png', show_shapes=True)

# 加载模型
from keras.models import load_model

model = load_model('model_House_MLP.h5')

# 预测
y_new = model.predict(x_test)
# 反归一化
min_max_scaler.fit(y_test_pd)
y_new = min_max_scaler.inverse_transform(y_new)


