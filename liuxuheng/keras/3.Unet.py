# ----------------开发者信息-------------------------------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月10日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息-------------------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate, ReLU
from keras.models import *
import keras
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、定义网络模型 -------------------------------
# 定义 3x3卷积
def conv3x3(out):
    return Conv2D(out, (3, 3), padding="same")
# 定义 3x3卷积 + ReLU
def ConvRelu(out):
    model = Sequential()
    model.add(conv3x3(out))
    model.add(ReLU())
    return model
# 定义 3x3卷积 + ReLU + 3x3卷积 + ReLU
def ConvRelu2(out):
    model = Sequential()
    model.add(conv3x3(out))
    model.add(ReLU())
    model.add(conv3x3(out))
    model.add(ReLU())
    return model
# 定义 上采样 + 3x3卷积
def UpConv(out):
    model = Sequential()
    model.add(UpSampling2D(size=(2, 2)))
    model.add(conv3x3(out))
    return model


input = Input(shape=(128, 128, 3))
# 第一部分，Unet,生成特征图Gatt
# 第一层
conv1 = ConvRelu2(64)(input)  # 128,128,64
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)  # 64,64,64
# 第二层
conv2 = ConvRelu2(128)(pool1)  # 64,64,128
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)  # 32,32,128
# 第三层
conv3 = ConvRelu2(256)(pool2)  # 32,32,256

# 第三层
up1 = UpConv(128)(conv3)  # 64,64,128
up1 = concatenate([up1, conv2], axis=3)  # 64,64,256
deconv1 = ConvRelu2(128)(up1)  # 64,64,128
# 第二层
up2 = UpConv(64)(deconv1)  # 128,128,64
up2 = concatenate([up2, conv1],axis=3)  # 128,128,128
deconv2 = ConvRelu2(64)(up2)  # 128,128,64
# 第一层
output1 = Conv2D(1, (1, 1), padding="same")(deconv2)  # 128, 128, 1


# 第二部分，提取特征部分，生成特征图Gft
conv11 = Conv2D(1, (9, 9), padding="same")(input)  # 128,128,1
output2 = Conv2D(1, (3, 3), padding="same")(conv11)  # 128,128,1
# Gft点乘Gatt
output3 = keras.layers.Multiply()([output1, output2])  # 128,128,1


# 第三部分，重构部分
conv111 = Conv2D(32, (3, 3), padding="same")(output3)
relu11 = ReLU()(conv111)
conv222 = Conv2D(3, (1, 1))(relu11)
output4 = MaxPooling2D(pool_size=(4, 4))(conv222)

model = Model(input, output4)
model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])
model.summary()
#  -------------------------- 2、定义网络模型 -------------------------------
