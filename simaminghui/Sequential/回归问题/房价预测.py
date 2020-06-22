# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/20 002017:00
# 文件名称：房价预测
# 开发工具：PyCharm


#  -------------------------- 导入需要包 -------------------------------
import keras
import numpy as np
from keras import models
from keras.layers import Dense, Dropout
from keras import regularizers  # 正则化
from sklearn.preprocessing import MinMaxScaler
import pandas as pd  # 数据预处理的工具

#   ---------------------- 数据的载入和处理 ----------------------------
path = 'D:\DataList\\boston\\boston_housing.npz'
f = np.load(path)
# 404个训练数据，102个测试数据
# 训练数据
x_train = f['x'][:404]  # 从下标0到403
y_train = f['y'][:404]
# 测试数据
x_valid = f['x'][404:]  # 从下标404到505
y_valid = f['y'][404:]
f.close()  # 关闭文件

# 转成DataFrame格式方便数据处理，DataFrame格式可理解为一张表,表格形式
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出房屋训练数据的x(前5个)
print('------------------')
print(y_train_pd.head(5))  # 输出房屋训练数据的y(前5个)

#  -------------------------- 数据归一化处理 -------------------------------
# 这里用的是线性归一化，公式是(x-xMin)/(xMax-xMin)
min_max_scale = MinMaxScaler()
min_max_scale.fit(x_train_pd)
x_train = min_max_scale.transform(x_train_pd)
min_max_scale.fit(y_train_pd)
y_train = min_max_scale.transform(y_train_pd)

# 测试集归一化
min_max_scale.fit(x_valid_pd)
x_valid = min_max_scale.transform(x_valid_pd)
min_max_scale.fit(y_train_pd)
y_valid = min_max_scale.transform(y_valid_pd)


#  -------------------------- 构建模型三种方法   -------------------------------


# ------------------------Sequential模型方法，最简单容易实现-------------------
def SequentialModel():
    global model
    model = models.Sequential()  # 初始化
    model.add(Dense(10,  # 输出大小，为模型加入一个13输入，10输出的全连接层，激活函数用relu
                    activation='relu',
                    input_shape=(x_train_pd.shape[1],)))

    model.add(Dropout(0.2))  # 丢弃神经元连接概率，防止过拟合，让20%的神经元不工作
    model.add(Dense(15,
                    kernel_regularizer=regularizers.l2(0.01),  # 施加在权重上的正则项  L2正则 根据公式添加上w绝对值之和,0.01是惩罚参数
                    activity_regularizer=regularizers.l1(0.01),  # 施加在权重上的正则项  L2正则 根据公式添加上w绝对值之和,0.01是惩罚参数
                    activation='relu',  # 激活函数
                    ))
    model.add(Dense(1,
                    activation='linear'  # 线性激活
                    )
              )
# ------------------------Sequential模型方法结束--------------------------------------------




# ---------------------------函数式API方法-------------------------------
# 相比于Sequential的单输入单输出，函数式API可以定义多个输入或输出

from keras.layers import Input
from keras.models import Model
def functionAPI():
    inputs = Input(shape=(x_train_pd.shape[1],))
    x = Dense(10, activation='relu')(inputs)
    x = Dropout(0.2)(x)
    x = Dense(15, activation='relu')(x)
    outputs = Dense(1)(x)
    global model
    model = Model(inputs=inputs, outputs=outputs)
# ---------------------------函数式API方法结束-------------------------------



# ------------------------------Model subclassing方法·····································
inputs = Input(shape=(x_train_pd.shape[1],))
class HouseModel(keras.Model):

    def __init__(self):
        super(HouseModel,self).__init__()
        self.dense1 = Dense(10,activation='relu')
        self.droput = Dropout(0.2)
        self.dense2 = Dense(15,activation='relu')
        self.dense3 = Dense(1)
    def call(self,inputs):
        x = self.dense1(inputs)
        x = self.droput(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

def SubclassingModel():
    global model
    model= HouseModel() # 模型实例化





# --------------------------------------------调用模型方法-------------------------------------

SequentialModel() # Sequential模型
# functionAPI()       #函数式API方法
# SubclassingModel()  # subclassing方法



# ---------------------------模型训练-------------------------------
model.compile(loss='mse',  # 损失均方差
              optimizer='adam',  # 优化器，优化loss
              metrics=['mae']  # 在训练过程中还监控一个新指标：平均觉得误差（MAE，mean absolute error）。它是预测值与目标值
              # 之差的绝对值。比如，如果这个问题的MAE等于0.5，就表示预测的房价与实际价格平均相差500美元。
              )
# 训练
history = model.fit(x_train, y_train,
                    epochs=100,  # 迭代次数，所有的数据训练200遍
                    batch_size=200,  # 每次用来梯度下降的批处理数据大小，200个数据一个批次
                    verbose=2,  # verbose:0：不输出训练过程，1：输出训练过程，2.输出每一个epoch
                    validation_data=(x_valid, y_valid)  # 验证集  每次epochs后都进行测试，可以提早发现问题，防止过拟合 超参数等等
                    )
print(model.summary())  # 打印网络层次结构
print(model.evaluate(x_valid, y_valid))

#  -------------------------- 模型可视化    ------------------------------
import matplotlib.pyplot as plt

# 绘制训练$验证的损失值
plt.plot(history.history['loss'])  # 训练集的loss
plt.plot(history.history['val_loss'])  # 测试集的loss
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
print(history.history.keys())
plt.plot(history.history['mae'], c='b')  # 训练集的mae
plt.plot(history.history['val_mae'], c='r')  # 测试集的mae
plt.show()
# 200轮得到mae为0.1348559558391571，由图像得epochs=100左右出现过拟合，将epochs改为100测试，mae为0.09689133614301682


#  -------------------------- 模型保存和预测 -----------------------------
from keras.utils import plot_model
from keras.models import load_model
# 保存模型
model.save('model_MLP.h5') # 创建一个HDF5的文件 ‘my_model.h5’

# 模型可视化,
# 如果报错（1）卸载pydot，
# （2）下载pydotplus，
# （3）将Python37\site-packages\keras\utils\vis_utils.py中pydot换为pydotplus（Ctrl+r替换）
# （4）安装GraphViz’s executables。Graphviz不是一个python tool，它是一个独立的软件http://www.graphviz.org/
# （5）配置环境（系统会默认配置，可以不用操作）
# （6）如果还是出现错误，重启软件或者电脑在进行尝试
plot_model(model,to_file='model_MLP.png',show_shapes=True)

# 加载模块
model = load_model('model_MLP.h5')

# 预测
y_new = model.predict(x_valid)
print(y_new)

# 反归一化
min_max_scale.fit(y_valid_pd)
y_new = min_max_scale.inverse_transform(y_new)
print(y_new)