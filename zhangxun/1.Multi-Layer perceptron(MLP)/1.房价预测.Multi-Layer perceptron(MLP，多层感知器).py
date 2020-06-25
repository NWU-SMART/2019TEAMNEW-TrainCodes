# ----------------------   代码布局： ---------------------- 
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测
# ----------------------   代码布局： ---------------------- 

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
#  -------------------------- 1、导入需要包 -------------------------------



#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
# 数据在服务器可以访问 
#train_data.shape:(404, 13),test_data.shape:(102, 13),
#train_targets.shape:(404,),test_targets.shape:(102,)
#the data compromises 13 features
#(x_train, y_train), (x_valid, y_valid) = boston_housing.load_data()  # 加载数据（国外服务器无法访问）

# 数据放到本地路径
# D:\\keras_datasets\\boston_housing.npz(本地路径)
path = 'E:/软件学习/深度学习/postgraduate study/数据集、模型、图片/1.MLP/boston_housing.npz'
f = np.load(path)
# 404个训练，102个测试
# 训练数据
x_train=f['x'][:404]  # 下标0到下标403
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 下标404到下标505
y_valid=f['y'][404:]
f.close()
# 数据放到本地路径

# 转成DataFrame格式方便数据处理
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
#  -------------------------- 3、数据归一化  -------------------------------

#  -------------------------- 4、模型训练   -------------------------------
# 单CPU or GPU版本，若有GPU则自动切换
model = Sequential()  # 初始化，很重要！
model.add(Dense(units = 10,   # 输出大小
                activation='relu',  # 激励函数
                input_shape=(x_train_pd.shape[1],)  # 输入大小, 也就是列的大小
               )
         )

model.add(Dropout(0.2))  # 丢弃神经元链接概率

model.add(Dense(units = 15,
#                 kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项  L2正则
#                 activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项 L1正则
                activation='relu' # 激励函数
                # bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
               )
         )

model.add(Dense(units = 1,   
                activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数  
               )
         )

print(model.summary())  # 打印网络层次结构

model.compile(loss='mse',  # 损失均方误差
              optimizer='adam',  # 优化器
             )

history = model.fit(x_train, y_train,
          epochs=200,  # 迭代次数
          batch_size=200,  # 每次用来梯度下降的批处理数据大小
          verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
          validation_data = (x_valid, y_valid)  # 验证集
        )

# 多GPU版本
# parallel_model = multi_gpu_model(model, gpus=4)
# parallel_model.compile(loss='mse',  # 多分类     
#                        optimizer='adam',
#                       )

# This `fit` call will be distributed on 4 GPUs.
# Since the batch size is 50, each GPU will process 32 samples.
# batch_size = 512
# epochs = 2
# history = parallel_model.fit(
#           x_train, 
#           y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split = 0.2  # 从训练集分割出20%的数据作为验证集
#         )
#  -------------------------- 4、模型训练    -------------------------------


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
# 保存模型
model.save('model_MLP.h5')  # creates a HDF5 file 'my_model.h5'

#模型可视化 pip install pydot
plot_model(model, to_file='model_MLP.png', show_shapes=True)

# 加载模型
model = load_model('model_MLP.h5')

# 预测
y_new = model.predict(x_valid)
# 反归一化
min_max_scaler.fit(y_valid_pd)
y_new = min_max_scaler.inverse_transform(y_new)
#  -------------------------- 6、模型保存和预测    ------------------------------
