# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.30
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/


# ----------------------代码布局-------------------#
# 1、读取数据同时进行数据预处理
# 2、定义dense block稠密块+transition layer 过渡块，并组成densenet网络
# 3、保存模型与模型可视化
# 4、训练测试

# ----------------------代码布局-------------------#
import numpy as np
import keras
import pickle
import matplotlib as mpl
mpl.use('Agg')#保证服务器可以显示图像

# ----------------------读取数据并对数据进行预处理-------------------#
# 分别通过数据地址读取数据1-5
data_batch_1 = pickle.load(open("cifar-10-batches-py/data_batch_1", 'rb'), encoding='bytes')
data_batch_2 = pickle.load(open("cifar-10-batches-py/data_batch_2", 'rb'), encoding='bytes')
data_batch_3 = pickle.load(open("cifar-10-batches-py/data_batch_3", 'rb'), encoding='bytes')
data_batch_4 = pickle.load(open("cifar-10-batches-py/data_batch_4", 'rb'), encoding='bytes')
data_batch_5 = pickle.load(open("cifar-10-batches-py/data_batch_5", 'rb'), encoding='bytes')

# 将数据划分为1-5的训练集
# 划分训练x
train_X_1 = data_batch_1[b'data']
# 将训练集转变类型，同时改变通道维度
train_X_1 = train_X_1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_1 = data_batch_1[b'labels']


train_X_2 = data_batch_2[b'data']
train_X_2 = train_X_2.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_2 = data_batch_2[b'labels']

train_X_3 = data_batch_3[b'data']
train_X_3 = train_X_3.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_3 = data_batch_3[b'labels']

train_X_4 = data_batch_4[b'data']
train_X_4 = train_X_4.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_4 = data_batch_4[b'labels']

train_X_5 = data_batch_5[b'data']
train_X_5 = train_X_5.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
train_Y_5 = data_batch_5[b'labels']


# 将训练数据堆叠，利用行扩展
'''
a = np.array([0, 1, 2])
b = np.array([3, 4, 5])
c = np.array([6, 7, 8])
np.row_stack((a, b, c))
array([[0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]])
'''
# 将训练的x数据进行堆叠
train_X = np.row_stack((train_X_1, train_X_2))
train_X = np.row_stack((train_X, train_X_3))
train_X = np.row_stack((train_X, train_X_4))
train_X = np.row_stack((train_X, train_X_5))

# 将训练的y数据进行堆叠
train_Y = np.row_stack((train_Y_1, train_Y_2))
train_Y = np.row_stack((train_Y, train_Y_3))
train_Y = np.row_stack((train_Y, train_Y_4))
train_Y = np.row_stack((train_Y, train_Y_5))


train_Y = train_Y.reshape(50000, 1).transpose(0, 1).astype("int32")
# print(train_Y.shape)
# 将训练y值进行one-hot编码
train_Y = keras.utils.to_categorical(train_Y)

# 读取测试集数据，并和训练集一样进行处理
test_batch = pickle.load(open("cifar-10-batches-py/test_batch", 'rb'), encoding='bytes')
test_X = test_batch[b'data']
test_X = test_X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
test_Y = test_batch[b'labels']
test_Y = keras.utils.to_categorical(test_Y)

# 对训练和测试数据进行归一化操作
train_X /= 255
test_X /= 255
# ----------------------读取数据并对数据进行预处理-------------------#


# ----------------------定义dense block稠密块+transition layer 过渡块，并组成densenet网络-------------------#
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
def Denselayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # 由于使用transpose(0, 2, 3, 1)，所以axis=3
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)

    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if drop_rate:
        x = Dropout(drop_rate)(x)
    return x
# 稠密块函数
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for i in range(nb_layers):
        conv = Denselayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x,conv],axis=-1)
    return x
# 过度块函数
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):


    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = BatchNormalization(axis=-1)(x)
    x = LeakyReLU(alpha=alpha)(x)
    # 卷积
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
#   通过is_max参数判断是否进行平均池化或最大池化
    if is_max != 0:
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)  # 最大池化
    else:
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)  # 平均池化

    return x


# 总体网络建设
growth_rate = 12

input = Input(shape=(32,32,3))

x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding='same')(input)
x = BatchNormalization(axis=-1)(x)
x = LeakyReLU(alpha=0.1)(x)
# 1层稠密块
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
# 2层稠密块
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
# 3层稠密块
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

x = BatchNormalization(axis=-1)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(units=10,activation='softmax')(x)

model = Model(input,x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='densenet.png',show_shapes=True)
# ----------------------定义dense block稠密块+transition layer 过渡块，并组成densenet网络-------------------#


# ----------------------训练参数-------------------#
import matplotlib.pyplot as plt
model.fit(train_X,train_Y,batch_size=200,epochs=10,verbose=1)
score = model.evaluate(test_X,test_Y,verbose=1)
plt.plot(score[0])
plt.savefig('loss.png')
plt.plot(score[1])
plt.savefig('accuracy.png')


