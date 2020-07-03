# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.7.2
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/
'''
unet是一个语义分割模型，其主要执行过程与其它语义分割模型类似，
首先利用卷积进行下采样，然后提取出一层又一层的特征，利用这一
层又一层的特征，其再进行上采样，最后得出一个每个像素点对应其
种类的图像
'''

# /------------------ 代码布局 --------------------*/
# 1、下采样函数downsampling_block
# 2、上采样函数upsampling_block
# 3、Unet模型函数unet
# /------------------代码布局 --------------------*/

# 下采样函数downsampling_block
from keras.layers import Conv2D,BatchNormalization,Activation,Dropout,MaxPool2D
def downsampling_block(input_tensor, filters, padding='valid',
                       batchnorm=False, dropout=0.0):
    # 一层卷积操作
    x = Conv2D(filters,kernel_size=(3,3),padding=padding)(input_tensor)
    if batchnorm:
     x = BatchNormalization()(x)
    else:
        x = x
    x = Activation('relu')(x)
    if dropout>0:
        x = Dropout(dropout)(x)
    else:
        x = x
    # 二层卷积操作
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    else:
        x = x
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    else:
        x = x
    return MaxPool2D(pool_size=(2,2))(x),x

from keras.layers import Conv2DTranspose,Cropping2D


# 上采样函数upsampling_block
from keras import backend as K
from keras.layers import Concatenate
def upsampling_block(input_tensor,skip_tensor, filters, padding='valid',
                       batchnorm=False, dropout=0.0):
    # 逆卷积操作
    x = Conv2DTranspose(filters,kernel_size=(3,3),strides=(2,2))(input_tensor)

    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:  # 使级联时像素大小一致
        cropping = ((h_crop // 2, h_crop - h_crop // 2), (w_crop // 2, w_crop - w_crop // 2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    x = Concatenate()([x, y])  # 特征级联


    # 一层卷积操作
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    else:
        x = x
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    else:
        x = x
    # 二层卷积操作
    x = Conv2D(filters, kernel_size=(3, 3), padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    else:
        x = x
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    else:
        x = x
    return x

from keras.models import Input,Model
from keras.layers import Lambda
# 定义Unet模型
def  Unet(height, width, channels, classes, features=64, depth=4,
         temperature=1.0, padding='valid', batchnorm=False, dropout=0.0):
    x = Input(shape=(height,width,channels))
    # 构建下采样过程
    input = x
    skip = []
    for i in range(depth):
        x,x0 = downsampling_block(x,features,padding,batchnorm,dropout)
        skip.append(x0)
        features *= 2

        # 一层卷积操作
    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    else:
        x = x
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    else:
        x = x
    # 二层卷积操作
    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    if batchnorm:
        x = BatchNormalization()(x)
    else:
        x = x
    x = Activation('relu')(x)
    if dropout > 0:
        x = Dropout(dropout)(x)
    else:
        x = x
#     构建上采样
#     反转迭代器
    for i in reversed(range(depth)):
        features //= 2
        x = upsampling_block(x, skip[i], features, padding,
                             batchnorm, dropout)
    x = Conv2D(filters=classes,kernel_size=(1,1))(x)
    # 利用lambda定义匿名函数使得z/temperature
    logits = Lambda(lambda z: z / temperature)(x)
    probabilities = Activation('softmax')(logits)

    return Model(inputs=input, outputs=probabilities)

model = Unet(256,256,3,20)
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='unet.png',show_shapes=True)



