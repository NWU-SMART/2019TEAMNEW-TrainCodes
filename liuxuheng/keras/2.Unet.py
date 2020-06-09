# ----------------开发者信息-------------------------------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月9日
# 修改日期：
# 修改人：
# 修改内容：
# 备注: 原文中网络的第一部分和第二部分均使用了resnet,这里没有使用，以后会加入
# ----------------开发者信息-------------------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate, ReLU
from keras.models import *
import keras
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、定义网络模型 -------------------------------
input = Input(shape=(128, 128, 3))
# 第一部分，Unet,生成特征图Gatt
# 第一层
conv1 = Conv2D(64, (3, 3), padding="same")(input)  # 128,128,64
relu1 = ReLU()(conv1)  # 128,128,64
conv1 = Conv2D(64, (3, 3), padding="same")(relu1)  # 128,128,64
relu2 = ReLU()(conv1)  # 128,128,64
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(relu2)  # 64,64,64
# 第二层
conv2 = Conv2D(128, (3, 3), padding="same")(pool1)  # 64,64,128
relu3 = ReLU()(conv2)  # 64,64,128
conv2 = Conv2D(128, (3, 3), padding="same")(relu3)  # 64,64,128
relu4 = ReLU()(conv2)  # 64,64,128
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(relu4)  # 32,32,128
# 第三层
conv3 = Conv2D(256, (3, 3), padding="same")(pool2)  # 32,32,256
relu5 = ReLU()(conv3)  # 32,32,256
conv3 = Conv2D(256, (3, 3), padding="same")(relu5)  # 32,32,256
relu6 = ReLU()(conv3)  # 32,32,256

# 第三层
up3 = UpSampling2D(size=(2, 2))(relu6)  # 64,64,256
up3 = Conv2D(128, (3, 3), padding="same")(up3)  # 64,64,128
up3 = concatenate([up3, relu4], axis=3)  # 64,64,256
deconv3 = Conv2D(128, (3, 3), padding="same")(up3)  # 64,64,128
relu7 = ReLU()(deconv3)  # 64,64,128
deconv3 = Conv2D(128, (3, 3), padding="same")(relu7)  # 64,64,128
relu8 = ReLU()(deconv3)  # 64,64,128
# 第二层
up2 = UpSampling2D(size=(2, 2))(relu8) # 128,128,128
up2 = Conv2D(64, (3, 3), padding="same")(up2)  # 128,128,64
up2 = concatenate([up2, relu2],axis=3) # 128,128,128
deconv2 = Conv2D(64, (3, 3), padding="same")(up2)  # 128,128,64
relu9 = ReLU()(deconv2)  # 128,128,64
deconv2 = Conv2D(64, (3, 3), padding="same")(relu9)  # 128,128,64
relu10 = ReLU()(deconv2)  # 128,128,64
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
