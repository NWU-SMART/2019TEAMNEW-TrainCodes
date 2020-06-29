#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.1.1 MLP-房价预测
#software：pycharm
#项目名称：房价预测（keras）


# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包  sss panda是一个可数据预处理的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练
# 5、训练可视化
# 6、模型保存和预测


# --------------------------导入包----------------------------#
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import numpy as np


# --------------------------载入数据集----------------------------#
path = 'D:\\keras\\房屋预测\\boston_housing.npz' #数据集地址#
f = np.load(path)      #numpy.load（）读取数据
# 404个训练，102个测试
# 训练集
x_train=f['x'][:404]  # 404个训练集（0-403）
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 102个测试集（404-505）
y_valid=f['y'][404:]
f.close()   # 关闭文件


# 转成DataFrame格式方便数据处理    DataFrame（panda）是二维的表格#
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出训练集的x (前5个)
print(y_train_pd.head(5))  # 输出训练集的y (前5个)


#  -------------------------- 3、数据归一化 -------------------------------
# MinMaxScaler：[ 0，1 ] ；MaxAbsScaler： [ -1，1 ]#
#训练集归一化#
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化#
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

#  -------------------------- 4、模型训练   -------------------------------


#  -------------------------- 1、Sequential()    -------------------------------
model = Sequential()
model.add(Dense(units = 10,   # 输入维10，输出为1的全连接层，relu激活
                activation='relu',
                input_shape=(x_train_pd.shape[1],)  # 输入大小, 也就是列的大小,总共13列  参数13*10+10

               )
         )

model.add(Dropout(0.2))    # 丢弃神经元链接概率20%#
model.add(Dense(units = 15,activation='relu'))
model.add(Dense(units = 1,    activation='linear'))  # 线性激励函数 #
print(model.summary())           # 打印网络层次结构#
model.compile(loss='mse',        # 损失均方误差
              optimizer='adam',  # 优化器，优化loss
              metrics=['acc']
             )                   #编译模型  keras model.compile(loss='目标函数 ', optimizer='adam', metrics=['accuracy'])#
#训练模型#
history = model.fit(x_train, y_train,
          epochs=200,            # 迭代次数200
          batch_size=200,        # 每次用来梯度下降的批处理数据大小 200个数据一个批次
          verbose=2,             # verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
          validation_data = (x_valid, y_valid),
          )
print(model.evaluate(x_valid,y_valid))

# -------------------------- 2.API类型--------------------------#
inputs=Input(shape=(x_train_pd.shape[1],))
x=Dense(10,activation='relu')(inputs)
x=Dense(15,activation='relu')(x)
x=Dense(1,activation='linear')(x)
predictions = Dense(1, activation='linear')(x)
model = Model(inputs=input, outputs=X)
model.compile(loss='MSE', optimizer='adam',metrics=['acc'])  #编译模型#
model.fit(x_train, y_train, epochs=200, batch_size=200)      #开始训练#

# -------------------------- 3.model类继承--------------------------#
class HousePredict(keras.Model):
    def __init__(self, use_dp=True):
        super(HousePredict, self).__init__(name='mlp')
        self.use_dp = use_dp                                     #进行dropout#
        self.dense1 = keras.layers.Dense(10, activation='relu')  # 输入为10，输出为1的全连接层#
        self.dense2 = keras.layers.Dense(15, activation='relu')  # 输入为15#
        self.dense3 = keras.layers.Dense(1, activation='linear') # 输入为1,线性激活#
        self.dp = keras.layers.Dropout(0.2)#dropout为0.2#

    def call(self, inputs):
        x = self.dense1(inputs)  #1
        if self.use_dp:
            x = self.dp(x)       #dropout
            #if self.use_bn:
            x=self.dense2(x)     #2
            x=self.dense3(x)     #3
    model = HousePredict()
    model.compile()              #编译模型#
    model.fit(x_train, y_train, epochs=200, batch_size=200)  #开始训练#

#  -------------------------- 模型可视化    ------------------------------#
import matplotlib.pyplot as plt         #matplotlib是python的2D绘图库
# 绘制训练集 & 测试集的损失值#
plt.plot(history.history['loss'])       #历史保留的训练集损失
plt.plot(history.history['val_loss'])   #历史保留的测试集损失
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['acc'],c='b')    #history保留的训练集accuracy
plt.plot(history.history['val_acc'],c='r')#history保留的测试集accuracy
plt.show()

#  -------------------------- 模型保存和预测    ------------------------------#
from keras.utils import plot_model
from keras.models import load_model

model.save('model_MLP.h5')  # creates a HDF5 file 'my_model.h5'# 保存模型
plot_model(model, to_file='model_MLP.png', show_shapes=True)#模型可视化 pip install pydot
model = load_model('model_MLP.h5')# 加载模型
y_new = model.predict(x_valid)#把x_valid放进网络，可以预测
min_max_scaler.fit(y_valid_pd)# 反归一化
y_new = min_max_scaler.inverse_transform(y_new)