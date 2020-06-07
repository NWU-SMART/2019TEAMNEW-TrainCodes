# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.5.24
#
# 版本号：Versoin 1.0
#
# 修改日期：2020.5.25
#
# 修改人：于林生
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# /------------------ 程序构成 --------------------*/
'''
1.导入需要的包
2.读取数据
3.数据预处理
4.建立模型
5.训练模型
6.结果显示
7.模型保存和预测
'''
# /------------------ 程序构成 --------------------*/


# /------------------导入需要的包--------------------*/
# keras预处理的包
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
# keras模型的包
from keras.models import Sequential
from keras.layers import Dense, Dropout,Input
from keras.models import Model
# 显示的包
import matplotlib.pyplot as plt
# /------------------导入需要的包--------------------*/

# /------------------读取数据--------------------*/
# 数据的路径
path = 'G:/python/code/多层感知器/boston_housing.npz'
# 读取路径的数据
data = np.load(path)
# 显示数据的索引
print(data.files) #['y','x']
# 根据索引查看大小
print(data['y'].shape)#(506,)
print(data['x'].shape)#(506,13)
# 划分数据集（测试集和验证集）80%为训练集20%为验证集
# 训练集
x_train = data['x'][:404]
y_train = data['y'][:404]
# 验证集
x_valid = data['x'][404:]
y_valid = data['y'][404:]
# 读取完了数据后关闭
data.close()
# /------------------读取数据--------------------*/

# /------------------数据预处理--------------------*/

# 转换类型，归一化处理
# 转成DataFrame格式方便数据处理
x_train = pd.DataFrame(x_train)#（404*13）
y_train = pd.DataFrame(y_train)#（404*1）
x_valid = pd.DataFrame(x_valid)#（102*13）
y_valid = pd.DataFrame(y_valid)#（102*1）
# 调用最大最小函数（缩放到0-1）
min_max_scaler = MinMaxScaler()
x_train = min_max_scaler.fit_transform(x_train)
x_valid = min_max_scaler.fit_transform(x_valid)
y_train = min_max_scaler.fit_transform(y_train)
y_valid = min_max_scaler.fit_transform(y_valid)

# /------------------数据预处理--------------------*/

# /------------------模型定义--------------------*/
#一种序贯模型（一条路走到黑）
# 缺点对于模型复杂的网络来说，不够灵活
# model = Sequential()
# model.add(Dense(units = 10,   # 输出大小
#                 activation='relu',  # 激励函数
#                 input_shape=(x_train.shape[1],)  # 输入大小, 也就是列的大小
#                )
#          )
# model.add(Dropout(0.2))
# model.add(Dense(units=15,
#                 activation='relu',
#                 input_dim=10,
#                 ))
# model.add(Dense(units=1,
#                 activation='relu',
#                 input_dim=15,
# ))

# /------------------模型定义--------------------*/
# 另一种序贯模型
# model = Sequential([
#     Dense(units=10,   # 输出大小
#         activation='relu',  # 激励函数
#         input_shape=(x_train.shape[1],)),  # 输入大小, 也就是列的大小
#     Dropout(0.2),
#     Dense(units=15,
#         activation='relu',
#         input_dim=10),
#     Dense(units=1,
#         activation='relu',
#         input_dim=15)
#
# ])

# /------------------模型定义--------------------*/
# 函数模型(类似torch)
# inputs = Input(shape=(x_train.shape[1],))
# x_train_1 = Dense(units=10,activation='relu')(inputs)
# x_train_1 = Dropout(0.2)(x_train_1)
# x_train_1 = Dense(units=15,activation='relu')(x_train_1)
# y_pred = Dense(units=1,activation='relu')(x_train_1)
# model = Model(inputs=inputs,outputs=y_pred)

# /------------------模型定义--------------------*/
# 类继承的方式
import keras
class simpleMLP(keras.Model):
    def __init__(self):
        super(simpleMLP,self).__init__(name='mlp')
        self.dense1 = keras.layers.Dense(units=10,activation='relu',input_shape=(x_train.shape[1],))
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(units=15,activation='relu')
        self.dense3 = keras.layers.Dense(units=1,activation='relu')
    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y_pred = self.dense3(x)
        return y_pred
model = simpleMLP()
# 定义损失函数和优化函数
model.compile(loss='mse',#均方损失
              optimizer='adam',
              )#adam优化
# 定义训练参数
epochs = 200
batch_size = 200


# /------------------模型定义--------------------*/

# /------------------模型训练--------------------*/

# 模型训练
predict=model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_valid,y_valid)
          )
# /------------------模型训练--------------------*/

# /------------------结果可视化--------------------*/

plt.plot(predict.history['loss'])
plt.plot(predict.history['val_loss'])
plt.title('Loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()

# /------------------结果可视化--------------------*/



#  -------------------------- 模型保存和预测    --------
from keras.utils import plot_model
from keras.models import load_model
# 保存模型
model.save_weights('weight.h5')
# model.save('多层感知器.h5')
# 加载模型
# model = load_model('多层感知器.h5')
model1 = Sequential([
    Dense(units=10,   # 输出大小
        activation='relu',  # 激励函数
        input_shape=(x_train.shape[1],)),  # 输入大小, 也就是列的大小
    Dropout(0.2),
    Dense(units=15,
        activation='relu',
        input_dim=10),
    Dense(units=1,
        activation='relu',
        input_dim=15)

])
model1.compile(loss='mse',#均方损失
              optimizer='adam',
              )#adam优化
model1.load_weights('weight.h5')
y_predict = model1.predict(x_valid)
# 查看预测值和真实值之间的差距
plt.plot(y_predict)
plt.plot(y_valid)
plt.show()
