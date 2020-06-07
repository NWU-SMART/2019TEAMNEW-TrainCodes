# ----------------开发者信息----------------------------
# 开发者：张涵毓
# 开发日期：2020年5月26日
# 内容：房价预测Class
# ----------------开发者信息----------------------------

# ----------------------代码布局-------------------------------------
# 1.引入keras，matplotlib，numpy，sklearn，pandas包
# 2.导入数据
# 3.数据归一化
# 4.模型建立
# 5.损失函数可视化
# 6.预测结果
# ---------------------------------------------------------------------

#---------------1、引入相关包--------------#
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout
from keras import regularizers
from keras import Model,Input
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#---------------1、引入相关包--------------#

#---------------2、导入数据---------------#
path='C:\\Users\\ZHB\\Desktop\\Keras代码\\1.Multi-Layer perceptron(MLP 多层感知器)\\boston_housing.npz' #数据路径
data = np.load(path)  # 读取路径数据
#404个训练 102个验证
#训练数据
train_x = data['x'][0:404]  # 前404个数据为训练集 0-403
train_y = data['y'][0:404]
#验证数据
valid_x = data['x'][404:]  # 后面的102个数据为测试集 404-505
valid_y = data['y'][404:]
data.close()

# 转成DataFrame格式方便数据处理
train_x_pd = pd.DataFrame(train_x)
train_y_pd = pd.DataFrame(train_y)
valid_x_pd = pd.DataFrame(valid_x)
valid_y_pd = pd.DataFrame(valid_y)

# 查看训练集前五条数据
print(train_x_pd.head(5))
print(train_y_pd.head(5))

#---------------2、导入数据---------------#

#  ------------ 3、数据归一化 ------------#
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(train_x_pd)
train_x = min_max_scaler.transform(train_x_pd)

min_max_scaler.fit(train_y_pd)
train_y = min_max_scaler.transform(train_y_pd)

# 验证集归一化
min_max_scaler.fit(valid_x_pd)
valid_x = min_max_scaler.transform(valid_x_pd)

min_max_scaler.fit(valid_y_pd)
valid_y = min_max_scaler.transform(valid_y_pd)
#  ------------ 3、数据归一化 ------------#

#  -------------4、Class模型建立--------------#
import keras
inputs=Input(shape=(train_x_pd.shape[1],))

class MLPhp(keras.Model):
    def __init__(self):
        super(MLPhp, self).__init__()
        self.dense1 = keras.layers.Dense(units=10,activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.dense2 = keras.layers.Dense(units=15,activation='relu')
        self.dense3 = keras.layers.Dense(units=1,activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

model = MLPhp()
print(model)  #网络可视化
model.compile(optimizer='adam',loss='mse')

#训练模型 存入损失值
result = model.fit(train_x, train_y, batch_size=200, epochs=200, verbose=2,
                   validation_data=(valid_x, valid_y))
                 #设置批量大小 迭代次数
#  -------------4、模型建立--------------#

# --------------5、损失函数可视化---------#
plt.plot(result.history['loss'])  # 从结果中获取训练集损失
plt.plot(result.history['val_loss'])  # 中获取验证集损失
plt.title('The Loss')
plt.xlabel('epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train', 'test'], loc='upper left')  # 曲线名字
plt.show()  # 画图
# --------------5、损失函数可视化---------#

# -------------6、保存模型预测结果--------#
model.save('HousePredict-sq.h5')  # 保存模型
plot_model(model, to_file='HousePredict-sq.jpg', show_shapes=True)  # 可视化模型
model = load_model('HousePredict-sq.h5')  # 加载模型

valid_pre = model.predict(valid_x)  # 用归一化后的数据进行预测

min_max_scaler.fit(valid_y_pd)  # 反归一化

pre= min_max_scale.inverse_transform(valid_pre)

# -------------6、保存模型预测结果--------#
