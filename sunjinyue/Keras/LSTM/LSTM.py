# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月16日
# 修改日期：
# 修改人：
# 修改内容：
'''

改自张亚楠代码

#  学习服务器上的LSTM程序是单层自编码器，所以我在网上上找了一个LSTM的示例数据集及信息，参考信息如下：
https://github.com/thenomemac/IMDB-LSTM-Tutorial
https://blog.csdn.net/Einstellung/article/details/82683652

#  此数据集是imdb的电影评论数据集，可以看到程序40-43行的输出结果，这个数据集已经事先做过单词转换为数字的处理
   不同的一串数字代表不同的单词，所以只需要将所有数据填充到同样长度再送入Embedding就可以了。
'''


#  -------------------------- 导入需要包 -------------------------------
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten,Embedding,LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.datasets import imdb

plt.style.use('ggplot') # 画的更好看


#  -------------------- 读取数据集及预处理 ---------------------
# 载入imdb电影评论数据集
# path = 'D:\\应用软件\\研究生学习\\imdb.npz'

(x_train,y_train),(x_test,y_test) = imdb.load_data(path='D:\\应用软件\\研究生学习\\imdb.npz',num_words=1000)  # sss 单词数量限制为1000  相对路径读取
# 定义规模

print(x_train.shape, y_train.shape)
print(x_train[0])  # 评论是已经处理好的数据，不同的数字代表不同的单词
print(y_train[0])  # 正面评价或负面评价 0或1

'''
输出结果：(25000,) (25000,)
[1, 14, 22, 16, 43, 530, 973, 2, 2, 65, 458, 2, 66, 2, 4, 173, 36, 256, 5, 
25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4,
 172, 112, 167, 2, 336, 385, 39, 4, 172, 2, 2, 17, 546, ·················]
 
 [1]
 
'''

# 把所有评论都填充到同样长度
max_len = 200
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)  # 填充到200
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)


#  ---------------------------------- 构建模型 --------------------------------------
model=Sequential()
# 用长度为16的向量表示这1000个单词,1000是输入维度，16是embedding维度，200是每个序列长度
model.add(Embedding(10000, 16, input_length=max_len))  # 用长度为16的向量表示这10000个单词,10000是输入维度，16是embedding维度，200是每个序列长度

# 添加LSTM层，含有128个隐藏单元
model.add(LSTM(units=128))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc']
)
# 记录训练过程
history = model.fit(x_train, y_train, epochs=5, batch_size=200, validation_data=(x_test, y_test))

print(model.evaluate(x_test, y_test))


#  --------------------- 训练过程可视化 ---------------------
# 画图
plt.plot(history.history.get('loss'), c='r')
plt.plot(history.history.get('val_loss'), c='b')
plt.show()
plt.plot(history.history.get('acc'), c='r')
plt.plot(history.history.get('val_acc'), c='b')
plt.show()
