# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/24
# 文件名称：预测房价.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入波士顿房价数据
# 3、数据归一化
# 4、K折验证
# 5、训练可视化
# 6、模型预测
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
from keras.datasets import boston_housing
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入波士顿房价数据------------------------------
#数据存在本地路径E:\\keras_datasets\\boston_housing.npz
path='E:\\keras_datasets\\boston_housing.npz'
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data(path)
#数据标准化
# 对每个特征做标准化，即对于输入数据的每个特征（输入数据矩阵中的列），减去特征平均值，再除以标准差，这样得到的特征平均值为0，标准差为1
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std
test_data-=mean
test_data /=std
#  -------------------------- 2、导入波士顿房价数据-------------------------------

#  -------------------------- 3、模型定义 -------------------------------
#由于样本数量很少，使用一个非常小的网络，包含两个隐藏层，每层有64个单元。
# 一般来说，训练数据越小，过拟合越严重，而较小的网络可以降低过拟合
#最后一层只有一个单元，没有激活，是一个线性层，这是标量回归的典型设置。添加激活函数将会限制输出范围
def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',#优化器
                  loss='mse', #均方误差MSE损失函数，预测值与目标值之差的平方，回归问题的常用损失函数
                  metrics=['mae'])#平均绝对误差MAE，预测值与目标值之差的绝对值
    return model

#  -------------------------- 3、模型定义 -------------------------------

#  -------------------------- 4、K折验证-------------------------------
#利用K折验证来验证使用的方法
'''
k=4
num_val_samples=len(train_data)//k
num_epochs=100
all_scores=[]
for i in range(k):
    print('processing fold #',i)
    val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]#准备验证数据：第k个分区的数据
    val_targets=train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate( # 准备训练数据：将其他所有分区的数据连起来
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)
    partial_train_targets=np.concatenate(
        [train_targets[:i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0)
    model=build_model()#构建Keras模型，已编译
    model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=2)#训练模型
    val_mse,val_mae=model.evaluate(val_data,val_targets,verbose=2)#在验证数据上评估模型
    all_scores.append(val_mae)
    all_scores #[1.9694947004318237, 2.567686080932617, 3.0149295330047607, 2.448160409927368]
    np.mean(all_scores) #2.5000676810741425  预测房价与实际价格平均相差2500美元，但实际价格范围在10000~50000美元，这一差别很大
'''
from keras import backend as K
K.clear_session()
    #保存每折的验证结果
num_epochs=500 #让训练轮次达到500
all_mae_histories=[]
for i in range(k):
    print('processing fold #', i)
    # 准备验证数据：第k个分区的数据
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    # 准备训练数据：其他所有分区的数据
partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)
partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
        train_targets[(i + 1) * num_val_samples:]],
        axis=0)
model = build_model()  # 构建Keras模型，已编译
history=model.fit(partial_train_data,
        partial_train_targets,
        validation_data=(val_data,val_targets),
        epochs=num_epochs,
        batch_size=1,
        verbose=2)  # 训练模型（静默模式，verbose=0）verbose：日志冗长度，int：冗长度，0：不输出训练过程，1：输出训练进度，2：输出每一个epoch
mae_history = history.history['val_mean_absolute_error']
all_mae_histories.append(mae_history)

#计算所有轮次中的K折验证分数平均值
 average_mae_history=[
    np.mean([x[i] for x in all_mae_histories])
    for i in range(num_epochs)]
 average_mae_history
#  -------------------------- 4、K折验证-------------------------------


#  -------------------------- 5、模型可视化-------------------------------
#绘制验证分数
plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
plt.xlabel=('Epochs')
plt.ylabel=('Validation MAE')
plt.show()

#绘制验证分数，删除前10个数据点
def smooth_curve(points,factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous=smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
        return smoothed_points
smooth_mae_history=smooth_curve(average_mae_history[10:])
plt.plot(range(1,len(smooth_mae_history)+1),smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
#  -------------------------- 5、模型可视化 -------------------------------

#  -------------------------- 6、模型预测 -------------------------------
model=build_model()
model.fit(train_data,train_targets,
          epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score=model.evaluate(test_data,test_targets)
test_mae_score #2.5532484335057877 预测的房价还是和实际价格相差约2550美元
#  -------------------------- 6、模型预测-------------------------------