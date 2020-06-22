# -*- coding: utf-8 -*-
# @Time: 2020/6/19 8:23
# @Author: wangshengkang
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import Input, add
from keras.models import Model
from keras.regularizers import l2  # 这里加入l2正则化目的是为了防止过拟合
from keras.utils.vis_utils import plot_model
import keras.backend as K
from keras.utils import plot_model


class ResNet:
    # 普通实例方法:第一个参数需要是self，它表示一个具体的实例本身。
    # 静态方法:如果用了装饰器@staticmethod，那么就可以无视这个self，而将这个方法当成一个普通的函数使用。
    # 类方法:而对于装饰器@classmethod，它的第一个参数不是self，是cls，它表示这个类本身。
    @staticmethod
    def residual_module(x, K, stride, chanDim, reduce=False, reg=1e-4, bnEps=2e-5, bnMom=0.9):
        #获取的结果为F(x)+x,输入的信号为x(直接跳转),F(x)为经过多个卷积处理函数
        shortcut = x
        #第一层卷积，归一激活卷积
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        act1 = Activation('relu')(bn1)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        # 第二层卷积，归一激活卷积
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding='same', use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # 第三层卷积，归一激活卷积
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if reduce:# 是否降维，如果降维的话，需要将stride设置为大于1,更改shortcut值
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        #计算F(x)+x
        x = add([conv3, shortcut])
        #函数返回F(x)+x
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset='cifar'):
        #初始化输入的高，宽，深度
        inputShape = (height, width, depth)
        chanDim = -1
        #如果通道维为第一位，改变顺序
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
        input = Input(shape=inputShape)#输入
        #在这里第一层使用BN层而不是使用conv，这样可以替代取平均值的操作
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(input)

        if dataset == 'cifar':#是否在CIFAR数据集上训练
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)

        #  ----- 1、按照层次数构建残差网络（多少个blok，每个block中有多少个残差块）
        # 构建多层残差网络（len(stages)）
        # stages=[3, 4, 6] --> 3个block，每个block中分别有3,4,6个残差块
        for i in range(0, len(stages)):# 每阶段的遍历
            # 构建第一个block，则设置步长为 (1,1)，否则为(2,2)
            stride = (1, 1) if i == 0 else (2, 2)

            #进行降维
            x = ResNet.residual_module(x, filters[i + 1], stride=stride, chanDim=chanDim, reduce=True, bnEps=bnEps,
                                       bnMom=bnMom)
            for j in range(0, stages[i] - 1):#每层的遍历
                #不进行降维
                x = ResNet.residual_module(x, filters[i + 1], stride=(1, 1), chanDim=chanDim, bnEps=bnEps, bnMom=bnMom)

        #------2构建全连接网络，进行分类
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)

        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)

        model = Model(input, x, name='ResNet')#构建模型

        model.summary()#打印模型结构

        return model

#调用resnet模型
model = ResNet.build(32, 32, 3, 10, stages=[3, 4, 6], filters=[64, 128, 256, 512])
#打印网络结构图
plot_model(model, to_file='resnet.png', show_shapes=True)
