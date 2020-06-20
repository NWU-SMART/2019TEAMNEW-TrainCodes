# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/19 00198:52
# 文件名称：波士顿房价
# 开发工具：PyCharm


# 预测20世纪70年代中期波士顿郊区房屋房价的中位数，已知当时郊区的一些数据点，比如犯罪率、房产税。该例子与前面两个有一个有趣的区别。它包含的数据点
# 相对较少，只有506个，分为404个训练样本和102个测试样本。输入数据的每个特征（比如犯罪率），都有不同的取值范围，有些特性是比例，取值范围
# 为0~1，有的取值范围为1~12，有的0~100

from keras.datasets import boston_housing

# 本地数据路径
path = 'D:\DataList\\boston\\boston_housing.npz'

# 404个训练样本，102个测试样本，每个样本有13个数值特性，如人均犯罪率，房间数，高速公路可达性，目标房屋价格单位是千美元
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data(path=path)
# 2 (404, 13) 404
print(train_data.ndim, train_data.shape, len(train_data))
# 1 (404,)
print(train_targets.ndim, train_targets.shape)

# 数据标准化，采用z-score标准化
mean = train_data.mean(axis=0)  # mean()作用求平均值，axis=0表示对各列求平均值,压缩行
# 1 (13,) 13
print(mean.ndim, mean.shape, len(mean))
train_data -= mean
std = train_data.std(axis=0)  # std()算标准差
train_data /= std
test_data -= mean
test_data /= std

# 构建网络，一般来说训练数据越少，过拟合越严重，而较小的网络可以降低过拟合
# 模型定义
from keras import models
from keras import layers


# 因为需要将同一个模型多次实例化，所以用一个函数构建模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    # 最后一层只有一个单元，没有激活，是一个线性层。这是标量回归（标量回归是预测单一连续的回归）的典型
    # 设置。添加激活函数将会限制输出范围。例如，如果最后一层使用sigmoid激活函数，网络智能学会预测
    # 0~1范围内的值，这里最后一层是纯线性的，所以网络可以学会预测任意范围内的值
    model.add(layers.Dense(1))
    # 此处编译用的mse损失函数，即均方误差（MSE，mean squared error），预测值与目标值只差的平方，
    # 这是回归函数常用的损失函数。
    # 在训练过程中还监控一个新指标：平均觉得误差（MAE，mean absolute error）。它是预测值与目标值
    # 之差的绝对值。比如，如果这个问题的MAE等于0.5，就表示预测的房价与实际价格平均相差500美元。
    model.compile(optimizer='rmsprop',
                  loss='mse',
                  metrics=['mae'])
    return model


# 训练数据较小，验证集的划分方式很可能会造成验证分数上有很大的方差，在这种情况下，最佳做法是使用L折交叉验证，这种方法
# 将可用数据划分为K个分区，实例化K个相同的模型，将每个模型在K-1个分区上训练，并在剩下一个分区上进行评估，模型验证分数等于K个验证分数的平均值
# K折验证
import numpy as np

# 设置k的取值
k = 4
num_val_samples = len(train_data) // k  # 每个分区的大小

# -------------------------------------------------------epochs=100------------------------------------------------------
# # 轮次为100
# num_epochs = 100
# all_scores = []
#
# for i in range(k):
#     print("验证分区是第", i, '个')
#     # 验证数据：第K个分区的数据
#     val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]
#     # 训练数据,concatenate()完成数组的拼接，axis=1表示对应的数组进行拼接，axis=0表示直接拼接
#     partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
#                                         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
#     # 构建keras模型
#     model = build_model()
#     # 开始训练,verbose = 0，在控制台没有任何输出,verbose = 1 ：显示进度条（注意： 默认为 1）verbose =2：为每个epoch输出一行记录
#     model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
#     # 得到均方误差，和平均绝对误差(预测值与目标值之差的绝对值)
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)
#
# # 全部结束后，[2.2306647300720215, 2.65234375, 2.7742013931274414, 2.510789394378662]，（参考结果，每次结果不同）
# print(all_scores)


# -------------------------------------------------------epochs=500------------------------------------------------------

# # 每次运行模型得到的验证分数有很大的差异，从2.6到3.2不等，平均分数（3.0）是比单一分数更可靠的指标——这就是K折交叉验证的关键，在这个例子中
# # 预测房价与实际价格平均相差3千美元，考虑到实际价格范围在10千~50千美元，这一差别还是很大
# # 让训练实际更长一点，达到500个轮次，记录模型在每轮的你表现，修改训练循环，保存每轮的验证分数记录
# # 500太耗时间，先测试100
# num_epochs = 100
# all_mae_histories = []
# for i in range(k):
#     print('processing fold:', i)
#     # 验证数据：第K个分区的数据
#     val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]
#     partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
#                                         axis=0)
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples],
#          train_targets[(i + 1) * num_val_samples:]], axis=0)
#     model = build_model()
#     history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
#                         epochs=num_epochs,
#                         batch_size=1)
#     print(history.history.keys())
#     mae_history = history.history["val_mae"]
#     all_mae_histories.append(mae_history)
# # 计算所有轮次中K折验证分数平均值
# print(all_mae_histories)
# # 计算所有轮次中的K折验证分数平均值
# average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
# print(len(average_mae_history), average_mae_history)


# 绘制验证分数
# -------------------------------------------------------验证可视化------------------------------------------------------

# import matplotlib.pyplot as plt
#
# plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
# plt.xlabel('Epochs')
# plt.ylabel('Validation MAE')
# plt.show()

# 根据图像显示大概80轮后不在显著降低，之后过拟合
# -------------------------------------------------------训练最终模型-----------------------------------------------------
# 训练最终模型
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)
