# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/29
#  @Author: MiJizong
#  @Content: MIMO——Keras
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: 基于多输入多输出的方法大多依赖API方式书写，简单快捷，直接堆叠层即可；
#            未解决使用类继承方法的输入接口的书写，暂时还没找到可以解决的方法。
# ----------------------开发者信息-----------------------------------------
# ----------------------   代码布局： -------------------------------------
# 1、导入相关的包
# 2、数据预处理
# 3、建立模型
# 4、模型显示与编译
# 5、模型预测
# ----------------------   代码布局： --------------------------------------
# --------------------  1、 导入相关的包 -----------------------------------

import keras
from keras import Input, Model, Sequential
from keras.layers import Dense, Concatenate, merge
import numpy as np
from keras.utils import plot_model
from numpy import random as rd

# --------------------  1、 导入相关的包 -----------------------------------
# --------------------  2、 数据预处理 -------------------------------------

samples_n = 3000
samples_dim_01 = 2
samples_dim_02 = 2
# 随机生成样本数据
x1 = rd.rand(samples_n, samples_dim_01)
x2 = rd.rand(samples_n, samples_dim_02)
y_1 = []
y_2 = []
y_3 = []
for x11, x22 in zip(x1, x2):  # zip() 打包元素为元组，返回由这些元组组成的列表
    # zip 方法在 Python 2 和 Python 3 中的不同：在 Python 3.x 中为了减少内存，zip() 返回的是一个对象。如需展示列表，需手动 list() 转换。
    y_1.append(np.sum(x11) + np.sum(x22))
    y_2.append(np.max([np.max(x11), np.max(x22)]))
    y_3.append(np.min([np.min(x11), np.min(x22)]))
y_1 = np.array(y_1)
y_1 = np.expand_dims(y_1, axis=1)  # 在1位置添加数据,，扩展数组的形状
y_2 = np.array(y_2)
y_2 = np.expand_dims(y_2, axis=1)
y_3 = np.array(y_3)
y_3 = np.expand_dims(y_3, axis=1)
# --------------------  2、 数据预处理 -------------------------------------

# --------------------  3、 模型建立 ---------------------------------------
# ********************Sequential方式********************
model1 = Sequential()
model2 = Sequential()
output_01 = Sequential()
output_02 = Sequential()
output_03 = Sequential()
# 全连接层
model1.add(Dense(units=3, name="dense_01", activation='softmax', input_shape=(samples_dim_01,)))
model1.add(Dense(units=3, name="dense_011", activation='softmax'))
model2.add(Dense(units=6, name="dense_02", activation='softmax', input_shape=(samples_dim_02,)))

# 加入合并层
output_01.add(Concatenate([model1.output, model2.output]))  # 尝试过以下表达，报错↓
output_02.add(Concatenate([model1.output, model2.output]))  # Concatenate()([model1.output, model2.output])
output_03.add(Concatenate([model1.output, model2.output]))  # (Concatenate()([model1, model2]))
# 分成两类输出 --- 输出01
output_01.add(Dense(units=6, activation="relu", name='output01'))
output_01.add(Dense(units=1, activation=None, name='output011'))
# 分成两类输出 --- 输出02
output_02.add(Dense(units=1, activation=None, name='output02'))
# 分成两类输出 --- 输出03
output_03.add(Dense(units=1, activation=None, name='output03'))

# ********************Sequential方式********************
# ********************API方式********************
# 输入层
inputs_01 = Input((samples_dim_01,), name='input_1')
inputs_02 = Input((samples_dim_02,), name='input_2')
# 全连接层
dense_01 = Dense(units=3, name="dense_01", activation='softmax')(inputs_01)
dense_011 = Dense(units=3, name="dense_011", activation='softmax')(dense_01)
dense_02 = Dense(units=6, name="dense_02", activation='softmax')(inputs_02)
# print("type:" .format(type(dense_011)))
# 加入合并层
merge = Concatenate()([dense_011, dense_02])
# 分成两类输出 --- 输出01
output_01 = Dense(units=6, activation="relu", name='output01')(merge)
output_011 = Dense(units=1, activation=None, name='output011')(output_01)
# 分成两类输出 --- 输出02
output_02 = Dense(units=1, activation=None, name='output02')(merge)
# 分成两类输出 --- 输出03
output_03 = Dense(units=1, activation=None, name='output03')(merge)
# 构造一个新模型
model_API = Model(inputs=[inputs_01, inputs_02], outputs=[output_011,
                                                      output_02,
                                                      output_03
                                                      ])

# ********************API方式********************
# *****************class继承方式*****************
class MIMO(keras.Model):
    def __init__(self,inputs1,inputs2):
        super(MIMO, self).__init__()

        # 全连接层
        self.dense1_1 = Dense(units=3, name="dense_01", activation='softmax', input_shape=(samples_dim_01,))
        self.dense1_2 = Dense(units=3, name="dense_011", activation='softmax')
        self.dense2_1 = Dense(units=6, name="dense_02", activation='softmax', input_shape=(samples_dim_02,))

        # 加入合并层
        # self.merge = keras.layers.merge.concatenate(
        #     [keras.layers.merge.concatenate([x1, x1]), x2])
        # ([self.dense1_1, self.dense1_2], self.dense2_1]))
        # self.merge = keras.layers.merge.concatenate([self.dense1_2,self.dense2_1])

        # 分成两类输出 --- 输出01
        self.output_01 = Dense(units=6, activation="relu", name='output01')
        self.output_011 = Dense(units=1, activation=None, name='output011')

        # 分成两类输出 --- 输出02
        self.output_02 = Dense(units=1, activation=None, name='output02')

        # 分成两类输出 --- 输出03
        self.output_03 = Dense(units=1, activation=None, name='output03')

    def call(self,inputs1,inputs2):
        x1 = self.dense1_1(inputs1)
        x1 = self.dense1_2(x1)
        x2 = self.dense2_1(inputs2)
        x_merge = keras.layers.merge.concatenate([x1, x2])
        output_01 = self.output_01(x_merge)
        output_011 = self.output_011(output_01)
        output_02 = self.output_02(x_merge)
        output_03 = self.output_03(x_merge)
        return output_011,output_02,output_03

model_class = MIMO()
# 未解决使用类继承方法的输入接口的书写，暂时还没找到可以解决的方法。

# *****************class继承方式*****************

# --------------------  3、 模型建立 ---------------------------------------

# --------------------  4、 模型显示与编译 ----------------------------------
# # 编译
# model.compile(optimizer="adam", loss='mean_squared_error', loss_weights=[1,
#                                     0.8,
#                                     0.8
#                                     ])


# 以下的方法可灵活设置
model_API.compile(optimizer='adam',
              loss={'output011': 'mean_squared_error',
                    'output02': 'mean_squared_error',
                    'output03': 'mean_squared_error'},
              loss_weights={'output011': 1,
                            'output02': 0.8,
                            'output03': 0.8})
# 训练
#model_class.fit([x1, x2], [y_1,y_2,y_3], epochs=50, batch_size=32, validation_split=0.1)

model_API.fit({'input_1': x1,
           'input_2': x2},
          {'output011': y_1,
           'output02': y_2,
           'output03': y_3},
          epochs=50, batch_size=32, validation_split=0.1)


# 显示模型情况
plot_model(model_API, show_shapes=True)
print(model_API.summary())
# --------------------  4、 模型显示与编译 ----------------------------------

# --------------------  5、 模型预测 ---------------------------------------
# 预测
test_x1 = rd.rand(1, 2)
test_x2 = rd.rand(1, 2)
test_y = model_API.predict(x=[test_x1, test_x2])
# 测试
print("测试结果：")
print("test_x1:", test_x1, "test_x2:", test_x2, "y:", test_y, np.sum(test_x1) + np.sum(test_x2))

# --------------------  5、 模型预测 ---------------------------------------
