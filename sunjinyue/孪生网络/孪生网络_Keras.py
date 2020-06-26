# ----------------开发者信息--------------------------------
# 开发者：孙进越
# 开发日期：2020年6月26日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------

# sss 通过函数调用model 和 直接调用model不一样 初步理解是前者不会权值共享每次都初始化 后者会权值共享  本次函数为不权值共享

#  -------------------------- 1、导入需要包 -------------------------------
from keras.models import Sequential
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
#  -------------------------- 1、导入需要包 -------------------------------


# ---------------------2、函数功能区-------------------------
def FeatureNetwork():
    """生成特征提取网络"""
    """这是根据，MNIST数据调整的网络结构，下面注释掉的部分是，原始的Matchnet网络中feature network结构"""

    # --- 输入数据
    inp = Input(shape = (28, 28, 1), name='FeatureNet_ImageInput')

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

    # ------------------------------ 网络第五层（全连接） --------------------------------------
    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    model = Model(inputs=inp, outputs=models)
    # ------------------------------ 网络第五层（全连接） -------------------------------------

    return model


def ClassiFilerNet():  # add classifier Net
    """生成度量网络和决策网络，其实maychnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)"""

    # ------------------- 构造两个孪生网络 --------------------
    input1 = FeatureNetwork()                     # 孪生网络中的一个特征提取
    input2 = FeatureNetwork()                     # 孪生网络中的另一个特征提取
    # ------------------- 构造两个孪生网络 --------------------

    # ------------------- 对于第二个网络各层更名 --------------------
    for layer in input2.layers:                   # 这个for循环一定要加，否则网络重名会出错。
        layer.name = layer.name + str("_2")
    # ------------------- 对于第二个网络各层更名 --------------------

    # ----- 两个网络的输入数据
    inp1 = input1.input
    inp2 = input2.input

    # ----- 两个网络层向量拼接
    merge_layers = concatenate([input1.output, input2.output])        # 进行融合，使用的是默认的sum，即简单的相加

    # ----- 全连接
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    # ----- 构建最终网络
    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])

    return class_models
# ---------------------2、函数功能区-------------------------


# ---------------------3、主调区-------------------------
matchnet = ClassiFilerNet()
matchnet.summary()  # 打印网络结构
plot_model(matchnet, to_file='D:/应用软件/研究生学习/model.png')  # 网络结构输出成png图片
# ---------------------3、主调区-------------------------
