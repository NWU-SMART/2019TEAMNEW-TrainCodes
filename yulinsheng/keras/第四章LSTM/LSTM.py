# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.6
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# /------------------ 程序构成 --------------------*/
'''
1.导入需要的包
2.读取数据
3.数据预处理
4.建立模型
5.训练模型
6.结果显示
7.模型保存和预测
'''
# /------------------ 程序构成 --------------------*/
# /------------------导入需要的包--------------------*/
import numpy as np
from keras.utils import np_utils
# /------------------数据预处理--------------------*/
# 生成随机数种子保证每次结果的代码相同
np.random.seed(10)
# 定义一个简单的数据
data = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# 将data编码
# 遍历data并将索引写入生成新的数据字典类型
char_to_int = dict((c, i) for i, c in enumerate(data))
int_to_char = dict((i, c) for i, c in enumerate(data))
# print(char_to_int)
# print(int_to_char)
# 构建数据
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(data) - seq_length, 1):
    # 定义前一个输入
    seq_in = data[i:i + seq_length]
    # 根据前一个的输入定义输出
    seq_out = data[i + seq_length]
    # 前一个数据的索引写入0-1,1-2这种的0写入dataX,1写入dataY
    dataX.append([char_to_int[char] for char in seq_in])
    # print(dataX)
    # 后一个数据的索引写入
    dataY.append(char_to_int[seq_out])
    # print(dataY)
    print(seq_in, '->', seq_out)
#     time steps
# LSTM处理的格式为[samples示例, time steps时间步数, features特征]
'''
LSTM输入和输出的理解
例如这样一个数据集合，总共100条句子，每个句子20个词，每个词都由一个80维的向量表示。
在lstm中，单个样本即单条句子输入下（shape是 [1 , 20, 80]）
keras中lstm的参数（samples， timestep， input_dim)
samples指批量训练样本的数量
timestep指一个句子输入多少个单词上述的timestep就是20
input_dim值代表一个timestep输入的维度
'''
input = np.reshape(dataX,(len(dataX),1,1))

# 将input归一化
input = input/float(len(data))
# 类别进行one-hot编码
y = np_utils.to_categorical(dataY)
print(y.shape)
# /------------------模型定义--------------------*/
# keras模型的包
from keras.models import Sequential
from keras.layers import Embedding,Dense, Dropout,LSTM
from keras.models import Model
# 序贯模型
Lstm = Sequential()
Lstm.add(LSTM(units=32,input_shape=(input.shape[1],input.shape[2])))
# Lstm.add(Dropout(0.5))
Lstm.add(Dense(units=y.shape[1],activation='softmax'))
# /------------------模型定义--------------------*/

# /------------------模型训练--------------------*/
# 模型训练
Lstm.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
result = Lstm.fit(input, y,epochs=500, batch_size=1, verbose=2)
# /------------------模型定义--------------------*/

# /------------------训练结果显示--------------------*/
import matplotlib.pyplot as plt

plt.plot(result.history['accuracy'],label='accuracy')
plt.plot(result.history['loss'],label='loss')
plt.show()
#
# # /------------------训练结果显示--------------------*/
#
# # /------------------训练结果预测--------------------*/
from keras.models import load_model
Lstm.save('LSTM.h5')
model = load_model('LSTM.h5')
# 用一个参数去预测下一个参数
test = np.reshape([[24]],(1,1,1))
# 预测下一个参数的索引
prediction = model.predict(test, verbose=0)
# 找到其中最大的索引
index = np.argmax(prediction)
# 最大索引对应的值
result = int_to_char[index]
print(result)
