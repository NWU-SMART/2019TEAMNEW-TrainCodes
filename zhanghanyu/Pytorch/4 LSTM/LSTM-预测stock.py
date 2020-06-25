# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月24日
# 内容：LSTM预测股价-pytorch
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#
# ------------------ 程序构成 --------------------*
'''
1.导入需要的包
2.加载stock数据
3.构造训练数据
4.构造LSTM
5.建模
6.预测stock
7.查看stock trend拟合效果
'''
# /------------------ 程序构成 --------------------*/
#----------------------1、导入需要的包----------------------------#
import quandl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
from torch import Tensor
import math
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

#--------------------4.构造LSTM-------------------------#
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # input gate  输入门的权重矩阵和bias矩阵
        self.w_ii = Parameter(Tensor(hidden_size, input_size))   # parameter是在定义函数function的时候，传到函数的参数：没有固定的值；
        self.w_hi = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ii = Parameter(Tensor(hidden_size, 1))
        self.b_hi = Parameter(Tensor(hidden_size, 1))

        # forget gate 遗忘门的权重矩阵和bias矩阵
        self.w_if = Parameter(Tensor(hidden_size, input_size))
        self.w_hf = Parameter(Tensor(hidden_size, hidden_size))
        self.b_if = Parameter(Tensor(hidden_size, 1))
        self.b_hf = Parameter(Tensor(hidden_size, 1))

        # output gate 输出门的权重矩阵和bias矩阵
        self.w_io = Parameter(Tensor(hidden_size, input_size))
        self.w_ho = Parameter(Tensor(hidden_size, hidden_size))
        self.b_io = Parameter(Tensor(hidden_size, 1))
        self.b_ho = Parameter(Tensor(hidden_size, 1))

        # cell  cell的的权重矩阵和bias矩阵
        self.w_ig = Parameter(Tensor(hidden_size, input_size))
        self.w_hg = Parameter(Tensor(hidden_size, hidden_size))
        self.b_ig = Parameter(Tensor(hidden_size, 1))
        self.b_hg = Parameter(Tensor(hidden_size, 1))
        self.reset_weigths()

    def reset_weigths(self):
        """reset weights
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, inputs: Tensor, state: tuple[Tensor]) \
            -> tuple[Tensor, tuple[Tensor, Tensor]]:   # ->用来提示该函数返回值的数据类型
        """Forward
        Args:
            inputs: [1, 1, input_size]
            state: ([1, 1, hidden_size], [1, 1, hidden_size])
        """
        #         seq_size, batch_size, _ = inputs.size()

        if state is None:
            h_t = torch.zeros(1, self.hidden_size).t()
            c_t = torch.zeros(1, self.hidden_size).t()
        else:
            (h, c) = state
            h_t = h.squeeze(0).t()
            c_t = c.squeeze(0).t()

        hidden_seq = []

        seq_size = 1
        for t in range(seq_size):
            x = inputs[:, t, :].t()
            # input gate
            i = torch.sigmoid(self.w_ii @ x + self.b_ii + self.w_hi @ h_t +
                              self.b_hi)
            # forget gate
            f = torch.sigmoid(self.w_if @ x + self.b_if + self.w_hf @ h_t +
                              self.b_hf)
            # cell
            g = torch.tanh(self.w_ig @ x + self.b_ig + self.w_hg @ h_t
                           + self.b_hg)
            # output gate
            o = torch.sigmoid(self.w_io @ x + self.b_io + self.w_ho @ h_t +
                              self.b_ho)

            c_next = f * c_t + i * g
            h_next = o * torch.tanh(c_next)
            c_next_t = c_next.t().unsqueeze(0)   # t（）矩阵转置
            h_next_t = h_next.t().unsqueeze(0)
            hidden_seq.append(h_next_t)

        hidden_seq = torch.cat(hidden_seq, dim=0)
        return hidden_seq, (h_next_t, c_next_t)

def reset_weigths(model):
    """reset weights
    """
    for weight in model.parameters():
        init.constant_(weight, 0.5)
#超参数
epochs=3
batch_size=16
#---------------------5、构造LSTM------------------------------------#
# #LSTM参数：return_sequences=True LSTM输出为一个序列。默认为Fulse，输出一个值。
#input_dim:输入单个样本特征值的维度
#input_length:输入的时间点长度
class Stock(nn.Module):
    def __init__(self):
        super (Stock,self).__init__()
        self.LSTM=LSTM(100,)
'''
model=Sequential()
model.add(LSTM(units=100,return_sequences=True,input_dim=x_train.shape[-1],input_length=x_train.shape[1]))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
'''
#---------------------6、预测stock价格----------------------------#
closing_price=Stock.predict(x_vaild)
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