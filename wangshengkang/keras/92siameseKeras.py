# -*- coding: utf-8 -*-
# @Time: 2020/6/18 12:04
# @Author: wangshengkang
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model


# 孪生网络相同的网络部分
def FeatureNetwork():
    inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')
    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)

    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    model = Model(inputs=inp, outputs=models)
    return model

# reuse为True，共享参数；为False，不共享参数
def ClassiFilerNet(reuse=False):
    if reuse:
        inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')
        models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
        models = Activation('relu')(models)
        models = MaxPool2D(pool_size=(3, 3))(models)

        models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
        models = Activation('relu')(models)

        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        models = Activation('relu')(models)

        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        models = Activation('relu')(models)

        models = Flatten()(models)
        models = Dense(512)(models)
        models = Activation('relu')(models)
        model = Model(inputs=inp, outputs=models)

        inp1 = Input(shape=(28, 28, 1))  # 创建输入
        inp2 = Input(shape=(28, 28, 1))  # 创建输入2
        model_1 = model(inp1)  # 孪生网络中的一个特征提取分支
        model_2 = model(inp2)  # 孪生网络中的另一个特征提取分支
        merge_layers = concatenate([model_1, model_2])  # 进行融合，使用的是默认的sum，即简单的相加

    else:
        input1 = FeatureNetwork()  # 孪生网络中的一个特征提取
        input2 = FeatureNetwork()  # 孪生网络中的另一个特征提取
        for layer in input2.layers:  # 这个for循环一定要加，否则网络重名会出错。
            layer.name = layer.name + str("_2")
        inp1 = input1.input
        inp2 = input2.input
        merge_layers = concatenate([input1.output, input2.output])  # 进行融合，使用的是默认的sum，即简单的相加
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])
    return class_models


matchnet = ClassiFilerNet(reuse=False)  # reuse为True，共享参数；为False，不共享参数
matchnet.summary()
plot_model(matchnet, to_file='modelpic.png')
