# ----------------开发者信息--------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年5月26日
# 开发框架：keras
#-----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、房价训练和测试数据载入
# 3、数据归一化
# 4、模型训练 （Sequential()类型的模型）
# 5、模型可视化
# 6、模型保存和预测
#--------------------------------------------------------------#

#  -------------------------- 1、导入需要包 -------------------------------
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#  -------------------------- 导入需要包 -------------------------------

#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
# 数据放到本地路径
path = 'E:\\study\\kedata\\boston_housing.npz'
f = np.load(path)
# 404个训练数据
x_train = f['x'][:404]   #训练数据下标0-403
y_train = f['y'][:404]   #训练标签下标0-403
# 102个验证数据
x_valid = f['x'][404:]   #验证数据下标404-505
y_valid = f['y'][404:]   #验证标签下标404-505
f.close()
# 数据放到本地路径

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(3))  # 输出训练数据的x (前3个)
print(y_train_pd.head(3))  # 输出训练数据的y (前3个)
# 转成DataFrame格式方便数据处理
#  -------------------------- 房价训练和测试数据载入 -------------------------------

#  -------------------------- 3、数据归一化 -------------------------------
# 归一化函数
min_max_scaler = MinMaxScaler()  # 归一到 [ 0，1 ]函数
# 训练集归一化
min_max_scaler.fit(x_train_pd) # 计算最大值最小值
x_train = min_max_scaler.transform(x_train_pd) # 归一化
min_max_scaler.fit(y_train_pd) # 计算最大值最小值
y_train = min_max_scaler.transform(y_train_pd) # 归一化
# 测试集归一化
min_max_scaler.fit(x_valid_pd) # 计算最大值最小值
x_valid = min_max_scaler.transform(x_valid_pd) # 归一化
min_max_scaler.fit(y_valid_pd) # 计算最大值最小值
y_valid = min_max_scaler.transform(y_valid_pd) # 归一化
#  -------------------------- 数据归一化 -------------------------------

#  -------------------------- 4、模型训练 （Sequential()类型的模型）  -------------------------------
model = Sequential() # 初始化
model.add(Dense(units = 10,    # 输出大小
                activation = 'relu', #激活函数
                input_shape = (x_valid_pd.shape[1],)  # 输入大小
                )
          )
model.add(Dropout(0.2))
model.add(Dense(units = 15,    # 输出大小
                activation = 'relu', #激活函数
                )
          )
model.add(Dense(units = 1,    # 输出大小
                activation = 'linear', #激活函数
                )
          )
print(model.summary())  # 打印网络层次结构
model.compile(loss='mse',  # 损失函数：均方误差
              optimizer='adam', # 优化器：adam
              )

history = model.fit(x_train, y_train, # 训练集
                    epochs=200,  # 迭代次数
                    batch_size=200,  # 每次用来梯度下降的批处理数据大小
                    verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                    validation_data  =  (x_valid, y_valid)  # 验证集
                   )
#  -------------------------- 模型训练 （Sequential()类型的模型）  -------------------------------

#  -------------------------- 5、模型可视化    ------------------------------
import matplotlib.pyplot as plt  # 导入包
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left') # loc:图例位置
plt.show()
#  -------------------------- 模型可视化    ------------------------------

#  -------------------------- 6、模型保存和预测    ------------------------------
# 导入包
from keras.utils import plot_model
from keras.models import load_model

model.save('model_MIP_HousePredict_sequential.h5') # 保存模型
plot_model(model, to_file='model_MIP_HousePredict_sequential.png', show_shapes=True) #模型可视化 pip install pydot
y_new = model.predict(x_valid) # 预测
# 反归一化
min_max_scaler.fit(y_valid_pd)
y_new = min_max_scaler.inverse_transform(y_new)
#  --------------------------模型保存和预测    ------------------------------