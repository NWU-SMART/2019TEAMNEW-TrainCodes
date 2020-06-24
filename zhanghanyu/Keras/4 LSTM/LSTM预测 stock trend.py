# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月23日
# 内容：LSTM预测股价
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#
# ------------------ 程序构成 --------------------*
'''
1.导入需要的包
2.加载stock数据
3.构造训练数据
4.LSTM建模
5.预测stock
6.查看stock trend拟合效果
'''
# /------------------ 程序构成 --------------------*/
#----------------------1、导入需要的包----------------------------#
import quandl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

#----------------------2、加载stock数据--------------------------#
start = data(2000,10,12)
end   = data.today()
google_stock=pd.DataFrame(quandl.get("WIKI/GOOGL",start_data=start,end_data=end))
print(google_stock.shape)
google_stock.tail()
google_stock.head()

#绘制stock历史收盘价trend图
plt.figure(figsize=(16,8))
plt.plot(google_stock['Close'])
plt.show()
#----------------------3.构造训练数据---------------------------#
#时间点长度
time_stamp=50

#划分训练集与验证集
google_stock=google_stock[['Open','High','Low','Close','Volume']]
train=google_stock[0:2800+time_stamp]
vaild=google_stock[2800-time_stamp:]
#归一化
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(train)
x_train,y_train=[],[]

#训练集
print(scaled_data.shape)
print(scaled_data[1,3])
for i in range(time_stamp,len(train)):
    x_train.append(scaled_data[i-time_stamp:i])
    y_train.append(scaled_data[i,3])

x_train,y_train=np.array(x_train),np.array(y_train)


#验证集
scaled_data=scaler.fit_transform(vaild)
x_vaild,y_vaild=[],[]
for i in range(time_stamp,len(vaild)):
    x_vaild.append(scaled_data[i-time_stamp:i])
    y_vaild.append(scaled_data[i,3])

x_vaild,y_vaild=np.array(x_vaild),np.array(y_vaild)
print(x_train.shape)
print(x_vaild.shape)
train.head()

#--------------------4.LSTM建模-------------------------#
#超参数
epochs=3
batch_size=16
#LSTM参数：return_sequences=True LSTM输出为一个序列。默认为Fulse，输出一个值。
#input_dim:输入单个样本特征值的维度
#input_length:输入的时间点长度
model=Sequential()
model.add(LSTM(units=100,return_sequences=True,input_dim=x_train.shape[-1],input_length=x_train.shape[1]))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
#---------------------5、预测stock价格----------------------------#
closing_price=model.predict(x_vaild)
scaler.fit_transform(pd.DataFrame(vaild['Close'].valie))
#反归一化
closing_price=scaler.inverse_transform(closing_price)
y_vaild=scaler.inverse_transform([y_vaild])
print(y_vaild)
print(closing_price)
rms=np.sqrt(np.mean(np.power((y_vaild-closing_price),2)))
print(rms)
print(closing_price.shape)
print(y_vaild.shape)
#------------------------6.查看stock trend拟合效果--------------------
plt.figure(figsize=(16,8))
dict_data={
    'Predictions':closing_price.reshape(1,-1)[0],
    'Close':y_vaild[0]
}
data_pd=pd.DataFrame(dict_data)

plt.plot(data_pd[['Close','Predictions']])
plt.show()