# -----------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月8日
# 内容:Unet
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras的包
# 2、下采样函数downsampling_block
# 3、上采样函数upsampling_block
# 4、Unet模型函数unet
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
from binstar_client.utils.projects.filters import filters
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import backend as K
#  ---------------------- 1、导入需要包 -----------------------------------
#-------------------------2、下采样函数downsampling_block -----------------
def downsampling_block(input_tensor,padding='valid',batchnorm=False,dropout=0.0):#下采样
    _,height,width,_=K.int_shape(input_tensor)#获取高度和宽度
    assert height%2==0
    assert width%2==0
    x=Conv2D(filters,kernel_size=(3,3),padding=padding)(input_tensor)
    x=BatchNormalization()(x) if batchnorm else x
    x=Activation('relu')(x)
    x=Dropout(dropout)(x) if dropout>0 else x
    x=Conv2D(filters,kernel_size=(3,3),padding=padding)(x)
    x=BatchNormalization()(x) if batchnorm else x
    x=Activation('relu')(x)
    x=Dropout(dropout)(x) if dropout>0 else x
    #两层卷积操作，提取特征
    return MaxPooling2D(pool_size=(2,2)),x
#-------------------------2、下采样函数downsampling_block -----------------
#-------------------------3、上采样函数upsampling_block -------------------
def upsampling_block(input_tensor,skip_tensor,padding='valid',batchnorm=False,dropout=0.0):#上采样
    x=Conv2DTranspose(filters,kernel_size=(2,2),strides=(2,2))(input_tensor)
    _,x_height,x_width,_=K.int_shape(x)
    _,s_height,s_width,_=K.int.shape(skip_tensor)
    h_crop=s_height-x_height
    w_crop=s_width-x_width
    assert h_crop>=0
    assert w_crop>=0
    if h_crop==0 and w_crop==0:
        y=skip_tensor
    else:
        cropping=((h_crop//2,h_crop-h_crop//2),(w_crop//2,w_crop-w_crop//2))#使级联时像素大小一致
        y=Cropping2D(cropping=cropping)(skip_tensor)
    x=Concatenate()([x,y])  #特征级联
    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x
    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x
    # --- 两层卷积操作 ------
    return x #返回dropout的值
#-------------------------3、上采样函数upsampling_block -------------------
#-------------------------4、Unet模型函数unet -----------------------------
def Unet(height, width, channels,classes,filters=64, depth=4, temperature=1.0, padding='valid', batchnorm=False, dropout=0.0):
    # 使用4个深度长的网络就是官网的典型网络
    x=Input(shape=(height,width,channels))#输入的特征
    inputs=x
    #构建下采样过程
    skip=[] #用于存放下采样中，每个深度后，dropout后的值，以供之后级联使用
    for i in range(depth):
        x,xo=downsampling_block(x,features,padding,batchnorm,dropout)
        skips.append(x0)
        features*=2  #下采样过程中，每个深度往下，特征翻倍，即每次使用翻倍数目的滤波器
    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters=features, kernel_size=(3, 3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x
    for i in range(range(depth)):
        features//=2  #每个深度往上。特征减少一倍
        x=upsampling_block(x,skip[i],features,padding,batchnorm,dropout)
    x=Conv2D(filters=classes,kernel_size=(1,1))(x)
    logits=Lambda(lambda z:z/temperature)(x) #简单的对x做一个变换
    probabilities=Activation('softmax')(logits) #对输出的两类做softmax，转换为概率。形式如【0.1,0.9],则预测为第二类的概率更大
    return Model(inputs=inputs,outputs=probabilities)
# -------------------------4、Unet模型函数unet -----------------------------
'''
它可以增加对输入图像的一些小扰动的鲁棒性，比如图像平移，旋转等，减少过拟合的风险，降低运算量，
和增加感受野的大小。升采样的最大的作用其实就是把抽象的特征再还原解码到原图的尺寸，最终得到分割结果。
通过数据增强使得有限且宝贵的的训练集利用的更加充分。
U型结构使定位准确，解决了医学图片的定位，而不是简单的二分类。
利用卷积层提取特征，获取每个像素点的信息，通过重叠结果，
可以完美对任意大小图分隔，也可以通过镜像图片，对图片的边界上的元素进行预测
'''





