# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.15
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/


# /------------------ 代码布局 --------------------*/
# 1.加载图像数据
# 2.图像数据预处理
# 3.给衣服赋予价格
# 4.对每张图像的价格进行归一化（去掉量纲，使数据具有可比性）
# 5.训练模型
# 6.保存模型与模型可视化
# 7.训练过程可视化
# /------------------ 代码布局 --------------------*/


# /-----------------  导入需要的包 --------------------*/
import numpy as np
import gzip
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像
import keras
from keras.preprocessing.image import ImageDataGenerator

# 由于图像数据比较大，使用显卡加速训练
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='3,4'
# 使用两块块显卡进行加速

# /-----------------  导入需要的包 --------------------*/

# /-----------------  读取数据 --------------------*/
# 写入文件路径
path_trainLabel = 'train-labels-idx1-ubyte.gz'
path_trainImage = 'train-images-idx3-ubyte.gz'
path_testLabel = 't10k-labels-idx1-ubyte.gz'
path_testImage = 't10k-images-idx3-ubyte.gz'
# 将文件解压并划分为数据集
with gzip.open(path_trainLabel,'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)#通过还原成类型ndarray。
with gzip.open(path_trainImage,'rb') as Imgpath:
    x_train = np.frombuffer(Imgpath.read(),np.uint8,offset=16).reshape(len(y_train),28,28,1)
with gzip.open(path_testLabel,'rb') as lbpath_test:
    y_test = np.frombuffer(lbpath_test.read(),np.uint8,offset=8)
with gzip.open(path_testImage, 'rb') as Imgpath_test:
    x_test = np.frombuffer(Imgpath_test.read(), np.uint8, offset=16).reshape(len(y_test),28,28,1)
# /-----------------  读取数据 --------------------*/
import cv2
# 输入数据维度是(60000, 28, 28)，
# 同时由于进行迁移学习时输入图片大小不能小于48*48所以将图片大小转换为48*48的
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test= [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
# 将数据转换类型，否则没有astype属性
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
# 将图片信息转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 归一化
x_train /= 255
x_test /= 255
# /-----------------  读取数据 --------------------*/

# /----------------- 伪造回归数据--------------------*/
import pandas as pd
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 给类别设置列名
y_train_pd.columns = ["label"]
y_test_pd.columns = ['label']
# 给每类衣服设置价格
value_mean = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]
# 利用正太分布给衣服生成价格
def set_price(row):
    # 利用上述的均值和3的标准差生成大小为1的价格，同时利用sort对价格排序
    price = sorted(np.random.normal(value_mean[int(row)],3,size=1))[0]
    # 保留两位小数，返回价格
    return np.round(price,2)
#  调用价格函数给训练数据写入价格
y_train_pd['price'] = y_train_pd['label'].apply(set_price)
y_test_pd['price'] = y_test_pd['label'].apply(set_price)
# 打印结果查看
print(y_train_pd.head(5))
print(y_test_pd.head(5))
'''
   label  price
0      9  28.73
1      0  45.30
2      0  47.56
3      3  96.45
4      0  44.87
   label   price
0      9   33.53
1      2   91.11
2      1   60.85
3      1   56.05
4      6  181.67
'''
# /----------------- 伪造回归数据--------------------*/

# /----------------- 数据归一化-------------------*/
from sklearn.preprocessing import MinMaxScaler
# 利用最大最小归一化
min_max = MinMaxScaler()
min_max.fit(y_train_pd)
# 将训练数据集中的price列进行归一化
y_trian = min_max.transform(y_train_pd)[:,1]
# 验证集归一化
min_max.fit(y_test_pd)
# 将训练数据集中的price列进行归一化
y_test = min_max.transform(y_test_pd)[:,1]
y_test_label = min_max.transform(y_test_pd)[:, 0]
# 归一化后数据类型转换为numpy.ndarray没有head
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
# print(y_train.head(5))
# print(y_test.head(5))
# /----------------- 数据归一化-------------------*/

# /----------------- 迁移学习模型建立-------------------*/
from keras import applications
from keras import Sequential
from keras.models import  Model
from keras.layers import Flatten,Dense,Dropout,Activation
base_model = applications.VGG16(include_top=False,
                                weights='imagenet',
                                input_shape=x_train.shape[1:])
model = Sequential()
# print(base_model.output)
# 将输出的结果转换为一维类型
# (7,7,512)->7*7*512
model.add(Flatten(input_shape=base_model.output.shape[1:]))
# 将输出维度转换为256
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.add(Activation('linear'))

model = Model(inputs=base_model.input,outputs=model(base_model.output))


# 保持模型的参数不变
for layer in model.layers[:15]:
    layer.trainable = False
opt = keras.optimizers.rmsprop(lr=1e-4,decay=1e-6)
model.compile(optimizer=opt,
              loss='mse')
# /----------------- 迁移学习模型建立-------------------*/

# /----------------- 模型训练-------------------*/
result = model.fit(x_train,y_train,
                   batch_size=32,epochs=5,
                   validation_data=(x_test,y_test),
                   shuffle=True)

import matplotlib.pyplot as plt
# 绘制训练 & 验证的损失值
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()



