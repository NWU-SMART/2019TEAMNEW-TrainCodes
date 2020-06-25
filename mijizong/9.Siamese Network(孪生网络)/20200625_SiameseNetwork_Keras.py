# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/6/25
#  @Author: MiJizong
#  @Content: SiameseNetwork——Keras
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: NULL
'''
孪生神经网络用于处理两个输入"比较类似"的情况。伪孪生神经网络适用于处理两个输入"有一定差别"的情况。
比如，我们要计算两个句子或者词汇的语义相似度，使用siamese network比较适合；如果验证标题与正文的
描述是否一致（标题和正文长度差别很大），或者文字是否描述了一幅图片（一个是图片，一个是文字），就
应该使用pseudo-siamese network。

三胞胎连体: 输入是三个，一个正例+两个负例，或者一个负例+两个正例，
Triplet在cifar, mnist的数据集上，效果超过了siamese network。
'''
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入需要包
# 2、函数功能区
# 3、主调区
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import keras
from keras.layers import Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
#  -------------------------- 1、导入需要包 -------------------------------

# -----------------------------2、函数功能区(API)--------------------------
def FeatureNetwork():
    """生成特征提取网络"""
    """这是根据，MNIST数据调整的网络结构，下面注释掉的部分是，原始的Matchnet网络中feature network结构"""

    # --- 输入数据
    inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')

    # ------------------------------ 网络第一层 --------------------------------------
    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)
    # ------------------------------ 网络第一层 --------------------------------------

    # ------------------------------ 网络第二层 --------------------------------------
    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
    models = Activation('relu')(models)
    # ------------------------------ 网络第二层 --------------------------------------

    # ------------------------------ 网络第三层 --------------------------------------
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    # ------------------------------ 网络第三层 --------------------------------------

    # ------------------------------ 网络第四层 --------------------------------------
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    # ------------------------------ 网络第四层 --------------------------------------

    # ------------------------------ 网络第五层（全连接） -----------------------------
    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    model = Model(inputs=inp, outputs=models)
    # ------------------------------ 网络第五层（全连接） -----------------------------

    return model

# -----尝试使用class类继承改写------
class FeatureNetwork_class(keras.Model):

    inputs_class = Input(shape=(28, 28,1))

    def __init__(self):
        super(FeatureNetwork_class,self).__init__()
        # 网络第一层
        self.conv1 = keras.layers.Conv2D(24,3,strides=1,padding='same',activation='relu')
        self.mp1 = keras.layers.MaxPooling2D(3)

        # 网络第二层
        self.conv2 = keras.layers.Conv2D(64,3,strides=1,padding='same',activation='relu')

        # 网络第三层
        self.conv3 = keras.layers.Conv2D(96,3,strides=1,padding='valid',activation='relu')

        # 网络第四层
        self.conv4 = keras.layers.Conv2D(96,3,strides=1,padding='valid',activation='relu')

        # 网络第五层
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(512,activation='relu')

    def call(self, inputs_class):
        x = self.conv1(inputs_class)
        x = self.mp1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

model_class =  FeatureNetwork_class()
# -----尝试使用class类继承改写------

def ClassiFilerNet():  # add classifier Net
    """生成度量网络和决策网络，其实matchnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)"""

    # ------------------- 构造两个孪生网络 --------------------
    input1 = FeatureNetwork()                     # 孪生网络中的一个特征提取
    input2 = FeatureNetwork()                     # 孪生网络中的另一个特征提取
    # ------------------- 构造两个孪生网络 --------------------

    # ------------------- 对于第二个网络各层更名 --------------
    for layer in input2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
        layer.name = layer.name + str("_2")
    # ------------------- 对于第二个网络各层更名 --------------

    # ----- 两个网络的输入数据
    inp1 = input1.input
    inp2 = input2.input
    #inp1 = Input(shape=(28, 28,1))
    #inp2 = Input(shape=(28, 28,1))

    # ----- 两个网络层向量拼接
    merge_layers = concatenate([input1.output, input2.output])  # 进行融合，使用的是默认的sum，即简单的相加

    # ----- 全连接
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    # ----- 构建最终网络
    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])

    return class_models
# -----------------------------2、函数功能区(API)--------------------------

# ------------------------------3、主调区----------------------------------

matchnet = ClassiFilerNet()
matchnet.summary()  # 打印网络结构
plot_model(matchnet, to_file='./model.png')  # 网络结构输出成png图片

# ------------------------------3、主调区----------------------------------