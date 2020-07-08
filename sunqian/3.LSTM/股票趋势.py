# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/
# 文件名称：
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、加载stock数据
# 3、构建并训练模型
# 4、模型训练及可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,LSTM
from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
import matplotlib.pyplot as plt

#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、加载stock数据 -------------------------------------------
# 读取数据
# look_back=1 指用前一个数据预测后一个数据 比如data是[1,2,3,4,5] look_back=1,那么x是[1,2,3] y就是[2,3,4]
def create_dataset(dataset,look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX),np.array(dataY)

stock_data=pd.read_csv('E:\keras_datasets\zgpa_train.csv',
                       header=0, parse_dates=[0],
                       index_col=0, usecols=[0,5],squeeze=True) # 只选择首尾两列
dataset = stock_data.values
print(stock_data.shape)
stock_data.head(10)

# 绘制股票历史趋势图
plt.figure(figsize=(12,8))
stock_data.plot()
plt.ylabel('Price')
plt.yticks(np.arange(0,300000000,100000000))
plt.show()

# 归一化
# feature_range=(0,1)指把所有数据标准化到(0,1)区间
scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset.reshape(-1,1))
# 划分训练集与测试集
train = int(len(dataset)*0.8)
test = len(dataset)-train
train,test =dataset[0:train],dataset[train:len(dataset)]

# 调用前面的函数分别生成训练集和测试集的x y
look_back =1
trainX, trainY = create_dataset(train,look_back)
testX, testY = create_dataset(test,look_back)
#  -------------------------- 2、加载stock数据--------------------------------------------

#  -------------------------- 3、构建并训练模型 -------------------------------------------
# 多层LSTM中当前层的input维度是上一层的output维度，return_sequences=True
epochs =3
batch_size = 16
model  = Sequential()
model.add(LSTM(units=100,return_sequences=True,input_dim=trainX.shape[-1],input_length=trainX.shape[1]))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size,verbose=1,validation_split=0.1)

#  -------------------------- 3、构建并训练模型 ------------------------------------------

#  -------------------------- 4、模型预测及可视化------------------------------------------
# 反归一化 让数据恢复原来的范围
trainpredict = model.predict(trainX)
testpredict = model.predict(testX)
trainpredict =scaler.inverse_transform(trainpredict)
trainY = scaler.inverse_transform(trainY)
testpredict = scaler.inverse_transform(testpredict)
testY = scaler.inverse_transform(testY)

# empty_like方法表示创建一个空数组，这个空数组很像dataset
trainpredictplot = np.empty_like(dataset)
trainpredictplot[:] = np.nan
trainpredictplot = np.reshape(trainpredictplot, (dataset.shape[0], 1))
# 下面操作相当于是一个100个数值的数组，填了前面70个，因为前面70个是训练集的预测值，后面30为空。
trainpredictplot[look_back: len(trainpredict)+look_back, :] = trainpredict

testpredictplot = np.empty_like(dataset)
testpredictplot[:] = np.nan
testpredictplot = np.reshape(testpredictplot, (dataset.shape[0], 1))
testpredictplot[len(trainpredict)+(look_back*2)+1: len(dataset)-1, :] = testpredict

#绘制图形
fig = plt.figure(figsize=(20, 15))
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainpredictplot)
plt.plot(testpredictplot)
plt.ylabel('price')
plt.xlabel('date')
plt.show()
# 蓝色是原始数据 黄色是训练数据，训练完进行预测，绿色是测试数据。可以看出大概趋势很符合，但有些地方的峰值预测不够
#  -------------------------- 4、模型预测及可视化------------------------------------------
