# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/18 00189:49
# 开发工具：PyCharm
# 电影评论分类：二分类
from keras.datasets import imdb

# 本地数据路径
path = 'D:\DataList\imdb\imdb.npz'
# 加载本地数据，labels是0和1的列表。0表示负面评论，1表示正面评论。data是一个向量。num_words=10000表示仅保留训练数据中前10000个最常出现的单词
# 测试和训练都是25000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path=path, num_words=10000)

# 将整数序列编码为二进制矩阵
import numpy as np


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))  # 创建一个len(sequences,dimension)的矩阵
    for i, sequence in enumerate(sequences):  # enumerate()可以得到索引和值
        results[i, sequence] = 1.  # 将results[i]的指定索引设为1
    return results


# 将训练数据和测试数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# 标签向量化
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# 构建网络
# 模型定义
from keras import models
from keras import layers

# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# # 模型编译，网络输出是一个概率值（网络最后一层使用sigmoid激活函数，仅包含一个单元），那么最好使用binary_crossentropy损失。
# # 对于输出概率值得模型，交叉熵crossentropy往往是最好的选择
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# # 留出验证集（为了在训练过程中监控模型在前所未见的数据上的精度，需要将原始训练数据留出10000个样本作为验证集）
# x_val = x_train[:10000]  # 验证数据
# partial_x_train = x_train[10000:]  # 训练数据
# y_val = y_train[:10000]  # 验证标签
# partial_y_train = y_train[10000:]  # 训练标签
#
# # 训练模型,model.ft返回一个History对象，这个对象有一个成员history，它是一个字典，包含训练过程中的所有数据，
# # history.history.keys(['val_loss', 'val_accuracy', 'loss', 'accuracy'])
# history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
# import matplotlib.pyplot as plt
#
# history_dict = history.history
# # 绘制训练损失和验证损失
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# print('我是loss_values', loss_values)
# epochs = range(1, len(loss_values) + 1)  # x轴的区间
# plt.plot(epochs, loss_values, 'bo', label='Training loss')  # 训练损失，bo表示蓝色圆点
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # 验证损失，b表示蓝色实线
# plt.title('Training and validation loss')  # 图像标题
# plt.xlabel('Epochs')  # x轴名称，训练次数
# plt.ylabel('loss')  # y轴名称
# plt.legend()  # 结束
# plt.show()  # 展示
#
# # 绘制训练精度和验证精度
# plt.clf()  # 清空图像
# acc = history_dict['accuracy']  # 训练精度
# val_acc = history_dict['val_accuracy']  # 验证精度
# print('我是acc:', acc)
# print(len(acc))
# epochs = range(1, len(acc) + 1)
# # 开始绘图
# plt.plot(epochs, acc, 'bo', label='Training acc')  # 训练的精度
# plt.plot(epochs, val_acc, 'b', label='Validation acc')  # 验证的精度
# plt.title('Training and validation acc')  # 图像标题
# plt.xlabel('Epochs')  # x轴名称，训练次数
# plt.ylabel('acc')  # y轴名称
# plt.legend()  # 结束
# plt.show()  # 展示


# 重新定义模型，训练4轮
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
# 预测,为正面的可能性大小
print(len(model.predict(x_test)))
print(model.predict(x_test))
