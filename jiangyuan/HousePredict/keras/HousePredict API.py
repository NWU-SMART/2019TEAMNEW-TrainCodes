# ----------------开发者信息--------------------------------#
# 开发者：姜媛
# 开发日期：2020年5月26日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#


# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包  sss panda是一个可数据预处理的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测
# ----------------------   代码布局： ----------------------


#  -------------------------- 1、导入需要包 -------------------------------
from keras.layers import Dense, Dropout
from keras.utils import multi_gpu_model
from keras import Model,Input  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
path = 'C:\\Users\\HP\\Desktop\\boston_housing.npz' # 数据路径
f = np.load(path) # numpy.load（）读取数据
# 404个训练，102个测试
# 训练数据
x_train=f['x'][:404]  # 下标0到下标403
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 下标404到下标505
y_valid=f['y'][404:]
f.close()   # 关闭文件
# 数据放到本地路径

# 转成DataFrame格式
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)
#  -------------------------- 2、房价训练和测试数据载入 -------------------------------


#  -------------------------- 3、数据归一化 -------------------------------
# 训练集归一化
min_max_scaler = MinMaxScaler()
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

#  -------------------------- 4、函数式API模型训练   -------------------------------
inputs = Input(shape=(x_train_pd.shape[1],)) # 输入
D1 = Dense(units=10, activation='relu')(inputs)
D2 = Dropout(0.2)(D1)
D3 = Dense(units=15, activation='relu')(D2)
Y = Dense(units=1, activation='linear')(D3) # 输出
model = Model(inputs=inputs, outputs=Y) # 模型输入输出
model.compile(loss='MSE', optimizer='adam') # 编译
model.summary() # 输出模型各层的参数状况
history=model.fit(x_train, y_train, epochs=200, batch_size=200, verbose=2, validation_data=(x_valid, y_valid))
#  -------------------------- 4、函数式API模型训练 -------------------------------


#  -------------------------- 5、模型可视化    ------------------------------
import matplotlib.pyplot as plt
# 绘制训练 and 验证的损失值
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
model.save('model_MLP.h5')  # creates a HDF5 file 'my_model.h5'# 保存模型
plot_model(model, to_file='model_MLP.png', show_shapes=True)
model = load_model('model_MLP.h5') # 加载模型
y_new = model.predict(x_valid) # 预测
min_max_scaler.fit(y_valid_pd) # 反归一化
y_new = min_max_scaler.inverse_transform(y_new)
#  -------------------------- 6、模型保存和预测    ------------------------------
