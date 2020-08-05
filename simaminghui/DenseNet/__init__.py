# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/8/4 000411:45
# 文件名称：__init__.py
# 开发工具：PyCharm
import pickle

from keras.layers import *
import numpy as np
import keras
# --------------------------------读入数据----------------------
# 加载数据集
data_batch_1 = pickle.load(open('D:\DataList\cifar-10-python\cifar-10-batches-py\data_batch_1','rb'),encoding='bytes')
data_batch_2 = pickle.load(open('D:\DataList\cifar-10-python\cifar-10-batches-py\data_batch_2','rb'),encoding='bytes')
data_batch_3 = pickle.load(open('D:\DataList\cifar-10-python\cifar-10-batches-py\data_batch_3','rb'),encoding='bytes')
data_batch_4 = pickle.load(open('D:\DataList\cifar-10-python\cifar-10-batches-py\data_batch_4','rb'),encoding='bytes')
data_batch_5 = pickle.load(open('D:\DataList\cifar-10-python\cifar-10-batches-py\data_batch_5','rb'),encoding='bytes')
# 数据转换
train_x_1 = data_batch_1[b'data']
train_x_1 = train_x_1.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_Y_1 = data_batch_1[b'labels']

train_x_2 = data_batch_2[b'data']
train_x_2 = train_x_2.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_Y_2 = data_batch_2[b'labels']

train_x_3 = data_batch_3[b'data']
train_x_3 = train_x_3.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_Y_3 = data_batch_3[b'labels']

train_x_4 = data_batch_4[b'data']
train_x_4 = train_x_4.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_Y_4 = data_batch_4[b'labels']

train_x_5 = data_batch_5[b'data']
train_x_5 = train_x_5.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_Y_5 = data_batch_5[b'labels']

# 拼接,row_stack(([1,2,3],[4,5,6]))==[[1,2,3],[4,5,6]]
train_X = np.row_stack((train_x_1,train_x_2))
train_X = np.row_stack((train_X,train_x_3))
train_X = np.row_stack((train_X,train_x_4))
train_X = np.row_stack((train_X,train_x_5))

train_Y = np.row_stack((train_Y_1,train_Y_2))
train_Y = np.row_stack((train_Y,train_Y_3))
train_Y = np.row_stack((train_Y,train_Y_4))
train_Y = np.row_stack((train_Y,train_Y_5))

# 此时train_Y.shape = （5,10000）
train_Y = train_Y.reshape(50000,1).transpose(0,1).astype("int32")
train_Y = keras.utils.to_categorical(train_Y)

# test
test_batch = pickle.load(open('D:\DataList\cifar-10-python\cifar-10-batches-py\\test_batch','rb'),encoding='bytes')
test_X = test_batch[b'data']
test_X = test_X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
test_Y = test_batch[b'labels']
print(test_Y)
test_Y = keras.utils.to_categorical(test_Y)

train_X /=255
test_X /=255






# ------------------------------建立稠密连接网络------------------------------------
# 稠密层函数
def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # 正则
    x = BatchNormalization(axis=3)(x)
    # 激活
    x = LeakyReLU(alpha=alpha)(x)
    # 卷积
    x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)

    # Composite function(组成函数)
    # 正则
    x = BatchNormalization(axis=3)(x)
    # 激活
    x = LeakyReLU(alpha=alpha)(x)
    # 卷积
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)

    if drop_rate:x = Dropout(drop_rate)(x)

    return x

# 稠密块函数
def DenseBlock(x,nb_layers,growth_rate,drop_rate=0.2):
     for i in range(nb_layers):
          conv = DenseLayer(x,nb_filter=growth_rate,drop_rate=drop_rate)
          # 拼接
          x = concatenate([x,conv],axis=3)
     return x

# 传输层函数
def TransitionLayer(x,comperssion=0.5,alpha=0.0,is_max=0):
     nb_filter = int(x.shape.as_list()[-1]*comperssion) # -1表示最后一个元素,缩小一半
     # 正则
     x = BatchNormalization(axis=3)(x)
     # 激活
     x = LeakyReLU(alpha=alpha)(x)
     # 卷积
     x = Conv2D(nb_filter,(1,1),strides=(1,1),padding="same")(x)
     if is_max!=0:
          x = MaxPooling2D(pool_size=(2,2),strides=2)(x) # 最大池化
     else:
          x = AveragePooling2D(pool_size=(2,2),strides=2)(x) # 平均池化

     return x

growth_rate = 12

# 输入数据维度
input = Input(shape=(32,32,3))

# 卷积
x = Conv2D(growth_rate*2,(3,3),strides=1,padding='same')(input)
# 正则
x = BatchNormalization(axis=3)(x)
# 激活
x = LeakyReLU(alpha=0.1)(x)

# ---3个稠密块-----

 # 第一个稠密块
x = DenseBlock(x,12,growth_rate,drop_rate=0.2)
 # 第一个传递层
x = TransitionLayer(x)

# 第二个稠密块
x = DenseBlock(x,12,growth_rate,drop_rate=0.2)
# 第二传递层
x = TransitionLayer(x)

# 第三个稠密块
x = DenseBlock(x,12,growth_rate,drop_rate=0.2)

# 正则
x = BatchNormalization(axis=3)(x)
# 平均池化
x = GlobalAveragePooling2D()(x)  # 可代替全连接层

# 10分类
x = Dense(10,activation="softmax")(x)


# 构建模型
import keras
model = keras.Model(input,x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 构建模型
# model.summary()


# ---计算Loss
for i in range(10):
    print('第{}次训练'.format(i+1))
    model.fit(train_X,train_Y,batch_size=100,epochs=1,verbose=1)
    score = model.evaluate(test_X,test_Y,verbose=1)
    print('test Loss =',score[0])
    print('test ACC =',score[1])

model.save("DenseNet.h5")
score = model.evaluate(test_X,test_Y,verbose=0)
print('test Loss =', score[0])
print('test ACC =', score[1])





