# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年5月26日
# 开发内容：房价预测keras框架的三种建造模型方法，添加了acc指标，画出来的图很奇怪，可能是回归问题没有acc这种说法
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
import keras
from keras.preprocessing import sequence
from keras.models import Sequential      # 顺序模型
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout     # 全连接层
from keras.utils import multi_gpu_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt  # 画图工具
import numpy as np    # 科学计算库
from sklearn.preprocessing import MinMaxScaler
import pandas as pd   # 数据预处理的工具



#   ---------------------- 数据的载入和处理 ----------------------------
path = 'boston_housing.npz'
f = np.load(path)      # numpy.load（）读取数据
# 404个训练，102个测试
# 训练数据
x_train = f['x'][:404]  # 下标0到下标403
y_train = f['y'][:404]
# 测试数据
x_valid = f['x'][404:]  # 下标404到下标505
y_valid = f['y'][404:]
f.close()   # 关闭文件

# 转成DataFrame格式方便数据处理    DataFrame格式可理解为一张表
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)



#  -------------------------- 数据归一化处理 -------------------------------
# 训练集归一化 归一化可以减少量纲不同带来的影响，使得不同特征之间具有可比性；
# 这里用的是线性归一化，公式是(x-xMin)/(xMax-xMin)
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


#  -------------------------- 构建模型三种方法   -------------------------------

# ------------------------Sequential模型方法，最简单容易实现-------------------
# model = Sequential()   #  初始化
# model.add(Dense(units = 10,   # 输出大小   为模型加入一个13输入，10输出的全连接层，激活函数用relu
#                activation='relu',  # 激活函数
#               input_shape=(x_train_pd.shape[1],)  # 输入大小, 也就是列的大小,总共13列  参数13*10+10
#
#               )
#         )
#
# model.add(Dropout(0.2))      #  丢弃神经元链接概率  防止过拟合,让20%的神经元不工作
#
# model.add(Dense(units = 15,
#                 kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项  L2正则 根据公式添加上w绝对值之和,0.01是惩罚参数
#                 activity_regularizer=regularizers.l1(0.01),  # 施加在输出上的正则项 L1正则  根据公式添加上w平方之和
#                activation='relu' # 激励函数
#                # bias_regularizer=keras.regularizers.l1_l2(0.01)  # 施加在偏置向量上的正则项
#              )
#         )
#
# model.add(Dense(units = 1,
#                activation='linear'  # 线性激励函数 回归一般在输出层用这个激励函数
#               )
#         )




# ---------------------------函数式API方法-------------------------------
# 相比于Sequential的单输入单输出，函数式API可以定义多个输入或输出，比如定义output1 output2
# from keras.layers import Input
# from keras.models import Model

# inputs = Input(shape=(x_train_pd.shape[1],))
# x = Dense(units=10, activation='relu')(inputs)  # 我的理解是把这一层看作了一个函数，参数是inputs
# x = Dropout(0.2)(x)
# x = Dense(units=15, activation='relu')(x)
# outputs = Dense(units=1, activation='linear')(x)
# model = Model(inputs=inputs, outputs=outputs)


# ------------------------------Model subclassing方法·····································
from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(x_train_pd.shape[1],))


class HouseModel(keras.Model):  # 继承keras.Model
    def __init__(self):   # 绑定属性
        super(HouseModel, self).__init__()
        self.dense1 = Dense(units=10, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(units=15, activation='relu')
        self.dense3 = Dense(units=1, activation='linear')

    def call(self, inputs):  # 模型调用的代码
        x = self.dense1(inputs)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


model = HouseModel()  # 实例化模型



# ---------------------------模型训练-------------------------------
model.compile(loss='mse',  # 损失均方误差       sss model.compile（） 加入优化函数
              optimizer='adam',  # 优化器，优化loss
              metrics=['acc']
              )   # 编译模型
# 训练
history = model.fit(x_train, y_train,    # sss 训练模型//训练过程
                    epochs=200,  # 迭代次数，所有的数据训练200遍
                    batch_size=200,  # 每次用来梯度下降的批处理数据大小 200个数据一个批次
                    verbose=2,  # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
                    validation_data=(x_valid, y_valid),  # 验证集   sss  每次epochs后都进行测试，可以提早发现问题，防止过拟合 超参数等等

                    )
print(model.summary())   # 打印网络层次结构
print(model.evaluate(x_valid, y_valid))



#  -------------------------- 模型可视化    ------------------------------
import matplotlib.pyplot as plt
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])  # history保留的训练集loss
plt.plot(history.history['val_loss'])  # history保留的测试集loss
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['acc'], c='b')    # history保留的训练集accuracy
plt.plot(history.history['val_acc'], c='r')  # history保留的测试集accuracy
plt.show()


#  -------------------------- 模型保存和预测 -----------------------------
from keras.utils import plot_model
from keras.models import load_model
# 保存模型
model.save('model_MLP.h5')  # creates a HDF5 file 'my_model.h5'

# 模型可视化 pip install pydot
plot_model(model, to_file='model_MLP.png', show_shapes=True)

# 加载模型
model = load_model('model_MLP.h5')

# 预测
y_new = model.predict(x_valid)  # 把x_valid放进网络，可以预测
# 反归一化
min_max_scaler.fit(y_valid_pd)
y_new = min_max_scaler.inverse_transform(y_new)


