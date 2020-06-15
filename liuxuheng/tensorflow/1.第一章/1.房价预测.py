# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月15日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
sess = tf.InteractiveSession()
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、数据载入 -------------------------------
# 数据存放路径
path = 'D:\\keras_datasets\\boston_housing.npz'
# 加载数据
f = np.load(path)
# 404个训练数据
x_train = f['x'][:404]  # 下标0到下标403
y_train = f['y'][:404]
f.close()
# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)
y_train_pd = pd.DataFrame(y_train)
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)
#  -------------------------- 2、数据载入 -------------------------------

#  -------------------------- 3、搭建模型   --------------------------------
in_units = 13  # 输入节点
h1_units = 10  # 隐层输出节点

W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))  # 隐含层的权重，初始化为截断的正态分布，其标准差为0.1
b1 = tf.Variable(tf.zeros([h1_units]))  # 隐含层的偏置，全部赋值为0

W2 = tf.Variable(tf.zeros([h1_units, 1]))  # 输出层权重
b2 = tf.Variable(tf.zeros([1]))  # 输出层的偏置

x = tf.placeholder(tf.float32, [None, in_units])  # 定义输入x的placeholder

keep_prob = tf.placeholder(tf.float32)  # #dropout的比率(即保留节点的概率)

hidden1 = tf.nn.relu(tf.matmul(x, W1)+b1)  # 定义一个激活函数为ReLU的隐含层hidden1

hidden1_drop = tf.nn.dropout(hidden1, keep_prob)  # 实现dropout功能，即随机将一部分节点置为0.其中keep_prob即为保留数据而不置为0的比例

y = tf.nn.softmax(tf.matmul(hidden1_drop, W2)+b2)  # 得到输出y

y_ = tf.placeholder(tf.float32, [None, 1])  # 定义输出y的placeholder
#  -------------------------- 3、搭建模型   --------------------------------

#  -------------------------- 4、训练模型   --------------------------------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y), reduction_indices=[1]))  # 交叉熵损失函数

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)  # 选择自适应优化器Adagrad，把学习速率设为0.3
tf.global_variables_initializer().run()

for i in range(10):
    batch_xs, batch_ys = x_train, y_train
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    print(y)
#  -------------------------- 4、训练模型   --------------------------------






