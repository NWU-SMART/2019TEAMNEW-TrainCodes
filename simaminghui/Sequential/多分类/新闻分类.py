# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/18 001814:01
# 文件名称：__init__.py
# 开发工具：PyCharm
# 新闻分类：多分类


# 构建一个网络，将路透社新闻划分为46个互斥的主题，因为有多个类别，所以这是多分类问题的一个例子。因为每个数据点只能划分
# 到一个类别，所以更具体的说，这是单标签，多分类问题的一个例子。如果每个数据点可以划分到多个类别，那他就是多标签，多分类。

# 加载数据集
from keras.datasets import reuters

# 加载本地数据路径
path = 'D:\DataList\\reuters\\reuters.npz'

# 训练数据有8982个，测试数据有2246个，num_words表示每个语句中的单词索引不超过10000,即每个向量中的数字没有大于10000
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(path=path, num_words=10000)
# 1 (8982,)，向量。和电影评论一样，每个向量的单词个数不同
print(train_data.ndim, train_data.shape)
# 样本标签是0~45范围内的整数，即话题索引编号，1 (8982,) [ 3  4  3 ... 25  3 25]
print(train_labels.ndim, train_labels.shape, train_labels)

import numpy as np


# 将数据向量化，定义函数
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 定义一个0矩阵
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)  # 将训练数据向量化
x_test = vectorize_sequences(test_data)  # 将测试数据向量化

from keras.utils.np_utils import to_categorical

# 标签向量化,将标签向量化有两种方法
# （1）将标签列表转换为整数张量
# （2）使用one-hot编码，one-hot编码是分类数据广泛使用的一种格式，也叫分类编码，标签的one-hot编码就是
# 将每个标签表示为全零向量，只有标签索引对应的元素为1。
one_hot_train_labels = to_categorical(train_labels)
print(one_hot_train_labels.ndim, one_hot_train_labels.shape, one_hot_train_labels[0])
one_hot_test_labels = to_categorical(test_labels)
# 构建网络
from keras import models
from keras import layers

# （1）网络最后一层是大小为46的Dense层，这意味着，对于每个输入样本，网络都会输出一个46维向量，这个向量的每个元素
# 代表不同的输出类别。
# （2）最后一层使用softmax激活，网络将输出在46个不同输出类别上的概率分布，其中output[i]是样本属于第i个类别的概率，
# 46个概率总和为1
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# 编译模型,对于这个例子最好的损失函数是categorical_crossentropy（分类交叉熵），
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 留出验证集
x_val = x_train[:1000]  # 验证数据
partial_x_train = x_train[1000:]  # 训练数据
y_val = one_hot_train_labels[:1000]  # 验证标签
partial_y_train = one_hot_train_labels[1000:]  # 训练标签

# 训练模型
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
import matplotlib.pyplot as plt


# 绘制训练损失和验证损失
# loss = history.history['loss']  # 训练数据的损失
# val_loss = history.history['val_loss']  # 验证数据的损失
# epochas = range(1, len(loss) + 1)  # x坐标区间
# plt.plot(epochas, loss, 'bo', label='Training loss')
# plt.plot(epochas, val_loss, 'b', label='Validation loss')
# plt.xlabel('Epochs')  # x轴的名称
# plt.ylabel('loss')  # y轴的名称
# plt.legend()  # 结束
# plt.show()  # 展示

# 绘制训练精度和验证进度
# acc = history.history['accuracy'] # 训练数据精度
# val_acc = history.history['val_accuracy'] # 验证数据精度
#
# epochs = range(1,len(acc)+1)
# plt.plot(epochs,acc,'bo',label='Training acc')
# plt.plot(epochs,val_acc,'b',label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('acc')
# plt.show()

# 大概8-9次过拟合,重新绘制
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,partial_y_train,epochs=8,batch_size=512,validation_data=(x_val,y_val))
print(model.evaluate(x_test,one_hot_test_labels))