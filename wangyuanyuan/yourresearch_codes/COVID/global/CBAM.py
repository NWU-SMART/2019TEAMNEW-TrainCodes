import numpy as ny
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from keras import backend as K


def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)  # 通道维度
    cbam_feature = spatial_attention(cbam_feature)  # 空间维度
    return cbam_feature


# 通道维度
def channel_attention(input_feature, ratio=8):
    # 获取当前的维度顺序
    print(type(input_feature))
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1  # 通道的轴

    channel = input_feature._keras_shape[channel_axis]  # shape是查看数据有多少行多少列

    # shareMLP-W0
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    # shareMLP-W1
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    # 经过全局平均池化
    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)  # assert断言操作，如果为false抛出异常

    # 经过W1
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel // ratio)
    # 经过W2
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool._keras_shape[1:] == (1, 1, channel)

    # 经过全局最大池化
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)
    # 经过W1
    max_pool = shared_layer_one(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel // ratio)
    # 经过W2
    max_pool = shared_layer_two(max_pool)
    assert max_pool._keras_shape[1:] == (1, 1, channel)

    # 逐元素相加，sigmoid激活
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    # 确定维度顺序
    if K.image_data_format() == "channels_first":  # channels_first(样本数，通道数，行，列)
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    C = multiply([input_feature, cbam_feature])

    # 逐元素相乘，得到下一步空间操作的输入feature
    return C


# 空间维度
def spatial_attention(input_feature):
    kernel_size = 7  # 卷积核7x7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)  # permute实现维度的任意顺序排列（或称置换）
    else:
        channel = input_feature._keras_shape[-1]  # 取最后一个元素（channel）
        cbam_feature = input_feature

    # 经过平均池化
    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # axis等于几，就理解成对那一维值进行压缩；二维矩阵（0为列1为行），x为四维矩阵（1,h,w,channel）所以axis=3，对矩阵每个元素求平均；
    # keepdims保持其矩阵二维特性
    assert avg_pool._keras_shape[-1] == 1  # 检验channel压缩为1

    # 经过最大池化
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1

    # 从channel维度进行拼接
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2  # 检验channel为2

    # 进行卷积操作
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    # 逐元素相乘，得到feature
    return multiply([input_feature, cbam_feature])