# ----------------开发者信息--------------------------------#
# 开发者：崇泽光
# 开发日期：2020年6月9日
# 修改日期：
# 修改人：
# 修改内容：
# --------------------------------------------------------#


# 导入所需要的包
import keras
from keras import Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# 读取数据
path = 'D:\\keras_datasets\\boston_housing.npz'
f = np.load(path)

# 设定训练集，测试集
x_train=f['x'][:404] # 下标0到下标403
y_train=f['y'][:404]
x_valid=f['x'][404:] # 下标404到下标505
y_valid=f['y'][404:]
f.close()

# 转换数据结构，方便数据处理（DataFrame）
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)

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

# 构建模型
inputs = Input(shape=(x_train_pd.shape[1],))

class MLP(keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.dense1 = keras.layers.Dense(10, activation='relu')
        self.dp = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(15, activation='relu')
        self.dense3 = keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dp(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

model= MLP()

#训练模型
model.compile(loss='mse',  # 损失（均方误差）
              optimizer='adam',  # 优化器
              )

history = model.fit(x_train, y_train,
          epochs=200,  # 迭代次数
          batch_size=200,  # 每次用来梯度下降的批处理数据大小
          verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
          validation_data = (x_valid, y_valid)  # 验证集
                    )

# 绘制训练和验证的损失值
import matplotlib.pyplot as plt
plt.plot(history.history['loss']) #调用plot函数在当前的绘图对象中绘图
plt.plot(history.history['val_loss'])
plt.title('Model loss') #设置图标的标题
plt.xlabel('Epoch') #设置x轴的文字
plt.ylabel('Loss') #设置y轴的文字
plt.legend(['Train', 'Test'], loc='upper left') #显示label中标记的图示
plt.show() #以上参数设置完毕后，使用plt.show()来显示出创建的所有绘图对象

