# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月24日
# 修改日期：
# 修改人：
# 修改内容：
'''
孪生网络是一种特殊类型的神经网络架构。与一个学习对其输入进行分类的模型不同，该神经网络是学习在两个输入中进行区分。它学习了两个输入之间的相似之处。
孪生网络由两个完全相同的神经网络组成，每个都采用两个输入图像中的一个。然后将两个网络的最后一层馈送到对比损失函数，用来计算两个图像之间的相似度。
它具有两个姐妹网络，它们是具有完全相同权重的相同神经网络。图像对中的每个图像将被馈送到这些网络中的一个。使用对比损失函数优化网络（我们将获得确切的函数）。
'''


#  -------------------------- 导入需要包 -------------------------------
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


# ---------------------函数功能区-------------------------
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
        layer.name = layer.name + str("_2")        # 用于防止网络层重名
    # ------------------- 对于第二个网络各层更名 --------------------

    # ----- 两个网络的输入数据
    inp1 = input1.input     # 第一个网络的输入
    inp2 = input2.input     # 第二个网络的输入

    # ----- 两个网络层向量拼接
    merge_layers = concatenate([input1.output, input2.output])        # 进行融合，使用的是默认的sum，即简单的相加

    # ----- 全连接
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    # ----- 构建最终网络
    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])     # 最终网络架构，特征层+全连接层

    return class_models

# ---------------------主调区-------------------------

matchnet = ClassiFilerNet()
matchnet.summary()  # 打印网络结构
plot_model(matchnet, to_file='G:/csdn攻略/picture/model.png')  # 网络结构输出成png图片

