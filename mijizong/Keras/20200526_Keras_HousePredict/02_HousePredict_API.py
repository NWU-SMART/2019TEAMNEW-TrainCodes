# ----------------------开发者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/26 20:02
# @Author: MiJizong
# @Version: 1.0
# @FileName: 1.0.py
# @Software: PyCharm
# ----------------------开发者信息-----------------------------------------


# ----------------------   代码布局： --------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测
# ----------------------   代码布局： --------------------------------------


#  -------------------------- 1、导入需要包 --------------------------------
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import regularizers, Model, Input  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

#  -------------------------- 1、导入需要包 --------------------------------


#  -------------------------- 2、房价训练和测试数据载入 ---------------------
path = 'D:\\Office_software\\PyCharm\\keras_datasets\\boston_housing.npz'
f = np.load(path)  # 读取上面路径的数据
# 404个数据用于训练，102个数据用于测试
# 训练数据
x_train = f['x'][:404]  # 下标0到下标403
y_train = f['y'][:404]
# 测试数据
x_valid = f['x'][404:]  # 下标404到下标505
y_valid = f['y'][404:]
f.close()  # 关闭文件

# 将数据转成DataFrame格式
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)
#  -------------------------- 2、房价训练和测试数据载入 ----------------------


#  ------------------------------ 3、数据归一化 -----------------------------
# 训练集归一化
min_max_scaler = MinMaxScaler()  # 归一化到 [ 0，1 ]
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)
#  -------------------------- 3、数据归一化  ------------------------------


#  -------------------------- 4、API模型训练   -------------------------------
inputs = Input(shape=(x_train_pd.shape[1],))
x = Dense(units=10, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(units=15, activation='relu')(x)

predictions = Dense(units=1, activation='linear')(x)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='MSE',                               # 损失均方误差
              optimizer='adam',                         # 优化器选择adam
              metrics=['accuracy'])                     # 评价函数使用accuracy   编译模型
model.summary()                                         # 输出各层的参数
history = model.fit(x_train, y_train, epochs=200,                 # 迭代200轮
          batch_size=200,                               # 每次用来梯度下降的批处理数据大小为200
          verbose=2,                                    # 日志冗长度为2
          validation_data=(x_valid, y_valid)            # 验证集
          )

#  -------------------------- 4、API模型训练    -------------------------------


#  -------------------------- 5、模型可视化    ------------------------------
import matplotlib.pyplot as plt

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#  -------------------------- 5、模型可视化    ------------------------------


#  -------------------------- 6、模型保存和预测    ------------------------------
from keras.utils import plot_model
from keras.models import load_model

model.save('model_MLP.h5')  # creates a HDF5 file 'my_model.h5'# 保存模型   需要pip install --upgrade h5py

plot_model(model, to_file='model_MLP.png', show_shapes=True)  # 模型可视化 需要pip install pydot
model = load_model('model_MLP.h5')  # 加载模型
y_new = model.predict(x_valid)  # 预测
min_max_scaler.fit(y_valid_pd)  # 反归一化
y_new = min_max_scaler.inverse_transform(y_new)
#  -------------------------- 6、模型保存和预测    ------------------------------
