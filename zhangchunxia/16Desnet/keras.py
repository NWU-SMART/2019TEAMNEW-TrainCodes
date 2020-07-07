# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月6日
# 内容:Desnet
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras的包
# 2、读取数据/数据预处理
# 3、搭建稠密连接网络模型
# 4、构建模型，保存模型-
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import pickle     # 用于python特有类型和python的数据类型间进行转换
import numpy as np
import keras
#  ---------------------- 1、导入需要包 -----------------------------------
#-------------------------2、读取数据/数据预处理----------------------------
'''
cifar数据集里面来自于80 million张小型图片的数据集，总数60000张，图片尺寸32*32，训练集50000张，测试集10000张
整个数据集被分为5个training batches和1个test batch。test batch：随机从每类选择10000张图片组成，
training batches：从剩下的图片中随机选择，但每类的图片不是平均分给batch的，总数为50000张图片，这些类别是完全互斥的。
'''
#   pickle.load 将pickle数据转换为python的数据结构
data_batch_1 = pickle.load(open("D:\\northwest\\小组视频\\keras_datasets\\cifar-10-batches-py\\data_batch_1", 'rb'),encoding='bytes')
#训练集的第一个batch，图片有10000张，picjle.load将每个batch文件转换为dictonary
data_batch_2 = pickle.load(open("D:\\northwest\\小组视频\\keras_datasets\\cifar-10-batches-py\\cifar-10-batches-py\\data_batch_2", 'rb'),encoding='bytes')
#训练集的第二个batch，图片有10000张
data_batch_3 = pickle.load(open("D:\\northwest\\小组视频\\keras_datasets\\cifar-10-batches-py\\cifar-10-batches-py\\data_batch_3", 'rb'),encoding='bytes')
#训练集的第三个batch，图片有10000张
data_batch_4 = pickle.load(open("D:\\northwest\\小组视频\\keras_datasets\\cifar-10-batches-py\cifar-10-batches-py\\data_batch_4", 'rb'),encoding='bytes')
#训练集的第四个batch，图片有10000张
data_batch_5 = pickle.load(open("D:\\northwest\\小组视频\\keras_datasets\\cifar-10-batches-py\\cifar-10-batches-py\\data_batch_5", 'rb'),encoding='bytes')
#训练集的第五个batch，图片有10000张

train_x_1 = data_batch_1[b'data']
train_x_1 = train_x_1.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")  # 将训练集转变类型，同时改变通道维度
train_y_1 = data_batch_1[b'labels']

train_x_2 = data_batch_2[b'data']
train_x_2 = train_x_2.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_y_2 = data_batch_2[b'labels']

train_x_3 = data_batch_3[b'data']
train_x_3 = train_x_3.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_y_3 = data_batch_3[b'labels']

train_x_4 = data_batch_4[b'data']
train_x_4 = train_x_4.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_y_4 = data_batch_4[b'labels']

train_x_5 = data_batch_1[b'data']
train_x_5 = train_x_5.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
train_y_5 = data_batch_5[b'labels']
# 矩阵的合并 行合并：np.row_stack()  列合并：np.column_stack()
# 训练数据堆叠可以使用行扩展
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

train_x = np.row_stack((train_x_1,train_x_2))    # 这一段的代码是将train_x 训练集合成一个大矩阵
train_x = np.row_stack((train_x,train_x_3))
train_x = np.row_stack((train_x,train_x_4))
train_x = np.row_stack((train_x,train_x_5))

train_y = np.row_stack((train_y_1,train_y_2))    # 这一段的代码是将train_y 训练集合成一个大矩阵
train_y = np.row_stack((train_y,train_y_3))
train_y = np.row_stack((train_y,train_y_4))
train_y = np.row_stack((train_y,train_y_5))

train_y = train_y.reshape(50000,1).transpose(0,1).astype("int32")  # .reshape 5000行1列 ，transport（1，0）表示行与列调换了位置
train_y = keras.utils.to_categorical(train_y)                      # 将整型的类别标签转为onehot编码

test_batch = pickle.load(open("D:\\northwest\\小组视频\\keras_datasets\\cifar-10-batches-py\\keras\\keras_datasets\\cifar-10-batches-py\\test_batch", 'rb'),encoding='bytes')
test_x = test_batch[b'data']
test_x = test_x.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
test_y = test_batch[b'labels']
test_y = keras.utils.to_categorical(test_y)        # 转为onehot编码

train_x /= 255    # 训练集归一化
test_x /= 255     # 测试集归一化
#-------------------------2、读取数据/数据预处理----------------------------
#-------------------------3、搭建稠密连接网络模型---------------------------
from keras.layers import Input,BatchNormalization,LeakyReLU,Conv2D,Dropout,concatenate,MaxPooling2D,AveragePooling2D,\
    GlobalAveragePooling2D,Dense
#定义dense block稠密块+transition layer 过渡块，并组成Densenet网络-------------------#
# 稠密层函数
def DenseLayer(x,nb_filter,bn_size=4,alpha=0.0,drop_rate=0.2):
    x = BatchNormalization(axis=3)(x)            # x（1000,32,32,3）这样的，axis=3 表示取通道正则，由于使用transpose(0, 2, 3, 1)，所以axis=3
    x = LeakyReLU(alpha=alpha)(x)                # 高级激活函数，alpha：x <0时激活函数的斜率.
    x = Conv2D(bn_size * nb_filter,(1,1),strides=(1,1),padding='same')(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)
    if drop_rate:
        x = Dropout(drop_rate)(x)
    return x
# 稠密块函数（通过连接稠密层函数）
def DenseBlock(x,nb_layers,growth_rate,drop_rate=0.2):
    # 构建nb_layers个稠密层
    for i in range(nb_layers):
        conv = DenseLayer(x,nb_filter=growth_rate,drop_rate=drop_rate)
        x = concatenate([x,conv],axis=3)   # 将x 和conv拼接在一起，循环nb_layers次，最后的输入是不断叠加的输入
        return x
# 传输层函数（稠密块和块之间的连接）
def TransitionLayer(x,compression=0.5,alpha=0.0,is_max=0):   # 为了减少维度，压缩率(compress rate)通常为0.5， 即减少一半的维度
    nb_filter = int(x.shape.as_list()[-1] * compression)     # x必须为tensor, x.shape()返回tuple,用as_list()转换为list
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if is_max != 0:
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
    else:
        x = AveragePooling2D(pool_size=(2,2),strides=2)(x)
    return x
#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------主函数部分------------------------------------------------------------#
growth_rate = 12                # 输出通道
input = Input(shape=(32,32,3))  # 输入
x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding='same')(input)
x = BatchNormalization(axis=3)(x)
x = LeakyReLU(alpha=0.1)(x)
# 3个稠密块 # 调用
x = DenseBlock(x,12,growth_rate,drop_rate=0.2) # 第一个稠密块
x = TransitionLayer(x)                         # 第一个传递层
x = DenseBlock(x,12,growth_rate,drop_rate=0.2) # 第二个稠密块
x = TransitionLayer(x)                         # 第二个传递层
x = DenseBlock(x,12,growth_rate,drop_rate=0.2) # 第三个稠密块
x = BatchNormalization(axis=3)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(10,activation='softmax')(x)          # 10分类
#-------------------------3、搭建稠密连接网络模型---------------------------
#-------------------------4、构建模型，保存模型-----------------------------
import matplotlib.pyplot as plt
from keras.models import Model,save_model,load_model
model = Model(input,x)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()                        # 打印模型
save_model(model,'DenseNet.h5')        # 保存模型
model = load_model('DenseNet.h5')      # 加载模型
from keras.utils import plot_model
plot_model(model, to_file="keras_DenseNet.png", show_shapes=True)  # 保存模型图片
# 计算loss
for i in range(10):
    print("Epoch:",i+1) # 因为是从0开始的
    model.fit(train_x,train_y,batch_size=100,epochs=1,verbose=1)
    score = model.evaluate(test_x,test_y,verbose=1)
    print('Test loss =',score[0])
    print('Test accuracy =', score[1])
# 输出结果
pred_y = model.predict(test_x)
score = model.evaluate(test_x,test_y,verbose=0)
print('Test loss =',score[0])
print('Test accuracy =', score[1])
plt.savefig('accuracy.png')
#-------------------------4、构建模型，保存模型-----------------------------