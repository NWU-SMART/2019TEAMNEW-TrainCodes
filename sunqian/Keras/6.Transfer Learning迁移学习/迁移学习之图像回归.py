# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/21
# 文件名称：迁移学习之图像回归.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入图像数据并预处理
# 3、迁移学习建模
# 4、模型训练和可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import gzip
import cv2
import numpy as np
import pandas as pd
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、 导入图像数据并预处理-------------------------------------------
# 写入文件路径
train_label_path = 'E:\\keras_datasets\\train-labels-idx1-ubyte.gz'
train_image_path = 'E:\\keras_datasets\\train-images-idx3-ubyte.gz'
test_label_path = 'E:\\keras_datasets\\t10k-labels-idx1-ubyte.gz'
test_image_path = 'E:\\keras_datasets\\t10k-images-idx3-ubyte.gz'

# 将文件解压并划分为数据集
with gzip.open(train_label_path, 'rb') as lbpath:
    y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)  # 还原成ndarray
with gzip.open(train_image_path, 'rb') as imgpath:
    x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
with gzip.open(test_label_path, 'rb') as lbpath_test:
    y_test = np.frombuffer(lbpath_test.read(), np.uint8, offset=8)
with gzip.open(test_image_path, 'rb') as imgpath_test:
    x_test = np.frombuffer(imgpath_test.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

# 图像预处理
# 输入数据的维度是(60000,28,28) 迁移学习时输入的图片大小不能小于48 * 48，vgg16 需要三维图像,所以扩充mnist的最后一维
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
# 转换数据类型为asarray 有astype属性
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

# 转换图片信息的数据类型
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
# 归一化
x_train /= 255.
x_test /= 255.

# --------------伪造回归数据------------------
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 给类别设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']
# 给每类衣服设置价格
value_mean = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]  # 均值列表
# 利用正态分布给衣服生成价格
def set_price(row):
    # 利用上述的均值和3的标准差生成大小为1的价格，同时利用sorted对价格排序
    price = sorted(np.random.normal(value_mean[int(row)], 3, size=1))[0]
    # 保留两位小数 返回价格
    return np.round(price, 2)
# 调用价格函数给训练数据写入价格
y_train_pd['price'] = y_train_pd['label'].apply(set_price)
y_test_pd['price']= y_test_pd['label'].apply(set_price)
# 查看结果
print(y_train_pd.head(5))
print(y_test_pd.head(5))
'''    label   price
 0      9   31.63
 1      0   44.35
 2      0   43.42
 3      3  104.19
 4      0   40.75
     label   price
 0      9   34.12
 1      2   80.16
 2      1   55.19
 3      1   54.21
4      6  175.60'''
# --------------伪造回归数据------------------

# --------------数据归一化--------------------
from sklearn.preprocessing import MinMaxScaler
# 利用最大最小归一化
min_max = MinMaxScaler()
min_max.fit(y_train_pd)
# 将训练数据集中的price列归一化
y_train = min_max.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max.fit(y_test_pd)
# 将验证数据集中的price和label列归一化
y_test = min_max.transform(y_test_pd)[:, 1]
y_test_label = min_max.transform(y_test_pd)[:, 0]
# 归一化后数据类型转换为numpy.ndarray 没有head
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

print(y_train.head(5))
print(y_test.head(5))
'''结果
         0
0  0.083747
1  0.160848
2  0.126166
3  0.356197
4  0.119749
          0
0  0.067114
1  0.315516
2  0.183863
3  0.141773
4  0.752795'''
# ---------------数据归一化-------------------
#  -------------------------- 2、导入图像数据并预处理--------------------------------------------

#  -------------------------- 3、迁移学习建模 -------------------------------------------
from keras import applications
from keras import Sequential
from keras.models import Model
from keras.layers import Flatten, Dense,Dropout, Activation
from keras.optimizers import RMSprop
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
model = Sequential()
print(base_model.output)  # 结果： Tensor("block5_pool/Identity:0", shape=(None, 1, 1, 512), dtype=float32)
# 将输出的结果转换为一维类型
model.add(Flatten(input_shape=base_model.output.shape[1:]))
# 将输出维度转换为256
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))
model.add(Activation('linear'))

model = Model(inputs=base_model.input, outputs=model(base_model.output))

# 保持模型的参数不变
for layer in model.layers[:15]:
    layer.trainable = False
opt = RMSprop(lr=1e-4, decay=1e-6)
model.compile(optimizer=opt, loss='mse')
#  -------------------------- 3、迁移学习建模 ------------------------------------------

#  -------------------------- 4、模型训练和可视化------------------------------------------
result = model.fit(x_train, y_train,
                   batch_size=32, epochs=5,
                   validation_data=(x_test, y_test),
                   shuffle=True)
import matplotlib.pyplot as plt
# 绘制训练和验证的损失值
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Train and Valid Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Transfer_Learning_valid_loss.png')
plt.show()
#  -------------------------- 4、模型训练和可视化------------------------------------------