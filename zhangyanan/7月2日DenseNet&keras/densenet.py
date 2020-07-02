# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年7月2日
# 修改日期：
# 修改人：
# 修改内容：

# ----------------------   代码布局： ----------------------
# 1、导入 Keras的包
# 2、读取数据
# 3、建立稠密连接网络模型
#      定义稠密层函数DenseLayer
#      定义稠密块函数DenseBlock
#      定义传输层函数TransitionLayer
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
import keras
from keras.models import Model, save_model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

## data
import pickle
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、读取数据 -------------------------------
data_batch_1 = pickle.load(open("cifar-10-batches-py/data_batch_1", 'rb'), encoding='bytes')
data_batch_2 = pickle.load(open("cifar-10-batches-py/data_batch_2", 'rb'), encoding='bytes')
data_batch_3 = pickle.load(open("cifar-10-batches-py/data_batch_3", 'rb'), encoding='bytes')
data_batch_4 = pickle.load(open("cifar-10-batches-py/data_batch_4", 'rb'), encoding='bytes')
data_batch_5 = pickle.load(open("cifar-10-batches-py/data_batch_5", 'rb'), encoding='bytes')

train_X_1 = data_batch_1[b'data']
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

train_X = np.row_stack((train_X_1, train_X_2))
train_X = np.row_stack((train_X, train_X_3))
train_X = np.row_stack((train_X, train_X_4))
train_X = np.row_stack((train_X, train_X_5))

train_Y = np.row_stack((train_Y_1, train_Y_2))
train_Y = np.row_stack((train_Y, train_Y_3))
train_Y = np.row_stack((train_Y, train_Y_4))
train_Y = np.row_stack((train_Y, train_Y_5))
train_Y = train_Y.reshape(50000, 1).transpose(0, 1).astype("int32")
train_Y = keras.utils.to_categorical(train_Y)

test_batch = pickle.load(open("cifar-10-batches-py/test_batch", 'rb'), encoding='bytes')
test_X = test_batch[b'data']
test_X = test_X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
test_Y = test_batch[b'labels']
test_Y = keras.utils.to_categorical(test_Y)

train_X /= 255
test_X /= 255
#  -------------------------- 2、读取数据 -------------------------------

#  -------------------------- 3、建立稠密连接网络模型 -------------------------------
  # 稠密层函数
def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # Bottleneck layers(瓶颈层)
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

    if drop_rate: x = Dropout(drop_rate)(x)

    return x

  # 稠密块函数
def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    # 构建nb_layers个稠密层
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        # 拼接
        x = concatenate([x, conv], axis=3)

    return x

  # 传输层函数
def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    nb_filter = int(x.shape.as_list()[-1] * compression)
      # 正则
    x = BatchNormalization(axis=3)(x)
      # 激活
    x = LeakyReLU(alpha=alpha)(x)
      # 卷积
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if is_max != 0:
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x) # 最大池化
    else:
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x) # 平均池化

    return x


growth_rate = 12

# 输入数据维度
inpt = Input(shape=(32, 32, 3))

# 卷积
x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding='same')(inpt)
# 正则
x = BatchNormalization(axis=3)(x)
# 激活
x = LeakyReLU(alpha=0.1)(x)

# --- 3个稠密块 -----

  # 第1个稠密块
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
  # 第1个传递层
x = TransitionLayer(x)

   # 第2个稠密块
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
   # 第2传递层
x = TransitionLayer(x)

   # 第3个稠密块
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
# --- 3个稠密块 -----

# 正则
x = BatchNormalization(axis=3)(x)
# 平均池化
x = GlobalAveragePooling2D()(x)
# 10分类
x = Dense(10, activation='softmax')(x)

#  -------------------------- 3、建立稠密连接网络模型 -------------------------------


# 构建模型
model = Model(inpt, x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 构建模型
model.summary()

# --- 计算Loss
for ii in range(10):
    print("Epoch:", ii + 1)
    model.fit(train_X, train_Y, batch_size=100, epochs=1, verbose=1)
    score = model.evaluate(test_X, test_Y, verbose=1)
    print('Test loss =', score[0])
    print('Test accuracy =', score[1])
# --- 计算Loss

# --- 保存模型
save_model(model, 'DenseNet.h5')
model = load_model('DenseNet.h5')

# --- 输出学习结果
pred_Y = model.predict(test_X)
score = model.evaluate(test_X, test_Y, verbose=0)
print('Test loss =', score[0])
print('Test accuracy =', score[1])

