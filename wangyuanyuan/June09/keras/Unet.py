#-------------------------------------------------------开发者信息-------------------------------------------------------
#开发者：王园园
#开发日期：2020.6.09
#开发软件：pycharm
#开发项目：U形网络

#---------------------------------------------------------代码布局-------------------------------------------------------
#1、导包
#2、下采样函数
#3、上采样函数
#4、Unet模型函数

#-----------------------------------------------------------导包--------------------------------------------------------
import keras.backend as K
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, Dropout, MaxPooling2D, Conv2DTranspose, Cropping2D, \
    Concatenate, Lambda

#------------------------------------------------------------下采样函数--------------------------------------------------
def downsampling_block(input_tensor, filters, padding='valid', batchnorm=False, dropout=0.0):
    # 获取高度和宽度
    _, height, width, _ = K.int_shape(input_tensor)
    assert height % 2 == 0
    assert width % 2 == 0

    #两层卷积操作
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x
    #返回的是池化后的值和dropout后的值，这里dropout后的值用于上采样特征级联
    return MaxPooling2D(pool_size=(2, 2))(x), x

#--------------------------------------------------------------上采样函数------------------------------------------------
def upsampling_block(input_tensor, skip_tensor, filters, padding='valid', batchnorm=False, dropout=0.0):
    x = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2))(input_tensor)
    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:
        cropping = ((h_crop//2, h_crop-h_crop//2), (w_crop//2, w_crop-w_crop//2))
        y = Cropping2D(cropping=cropping)(skip_tensor)
    x = Concatenate()([x, y])

    #两层卷积操作
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return x

#------------------------------------------------------------Unet模型---------------------------------------------------
#使用4个深度长的网络就是官网的典型网络
def unet(height, width, channels, classes, features=64, depth=4, temperature=1.0, padding='valid', batchnorm=False, dropout=0.0):
    x = Input(shape=(height, width, channels))
    inputs = x

    #构建下采样过程
    skips = []
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding, batchnorm, dropout)
        skips.append(x0)
        #下采样过程中，每个深度往下，特征翻倍，及每次使用翻倍数目的滤波器
        features *= 2

    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x
    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    for i in reversed(range(depth)):
        features // 2    #每个深度往上，特征减少一倍
        x = upsampling_block(x, skips[i], features, padding, batchnorm, dropout)

    x = Conv2D(filters=classes, kernel_size=(1, 1))(x)

    logits = Lambda(lambda z: z/temperature)(x)
    probabilities = Activation('softmax')(logits)

    return Model(inputs=inputs, outputs=probabilities)






