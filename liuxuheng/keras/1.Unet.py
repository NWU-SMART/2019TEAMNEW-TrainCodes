from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate
from keras.models import *
import keras

input = Input(shape=(512, 512, 3))

# 第一部分，Unet
# 第一层 输入 512,512,3, 输出 256,256,64
conv1 = Conv2D(64, (3, 3), padding="same")(input)
conv1 = Conv2D(64, (3, 3), padding="same")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
# 第二层 输入 256,256,64, 输出 128,128,128
conv2 = Conv2D(128, (3, 3), padding="same")(pool1)
conv2 = Conv2D(128, (3, 3), padding="same")(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
# 第三层 输入 128,128,128, 输出 64,64,256
conv3 = Conv2D(256, (3, 3), padding="same")(pool2)
conv3 = Conv2D(256, (3, 3), padding="same")(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
# 第四层 输入 64,64,256, 输出 32,32,512
conv4 = Conv2D(512, (3, 3), padding="same")(pool3)
conv4 = Conv2D(512, (3, 3), padding="same")(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv4)
# 第五层 输入 32,32,512, 输出 32,32,1024
conv5 = Conv2D(1024, (3, 3), padding="same")(pool4)
conv5 = Conv2D(1024, (3, 3), padding="same")(conv5)

# 第五层 上采样，卷积，通道拼接，卷积
up5 = UpSampling2D(size=(2, 2))(conv5)
up5 = Conv2D(512, (3, 3), padding="same")(up5)
up5 = concatenate([up5, conv4], axis=3)
deconv5 = Conv2D(512, (3, 3), padding="same")(up5)
deconv5 = Conv2D(512, (3, 3), padding="same")(deconv5)

# 第四层 上采样，卷积，通道拼接，卷积
up4 = UpSampling2D(size=(2,2))(deconv5)
up4 = Conv2D(256, (3, 3), padding="same")(up4)
up4 = concatenate([up4, conv3], axis=3)
deconv4 = Conv2D(256, (3, 3), padding="same")(up4)
deconv4 = Conv2D(256, (3, 3), padding="same")(deconv4)

# 第三层 上采样，卷积，通道拼接，卷积
up3 = UpSampling2D(size=(2,2))(deconv4)
up3 = Conv2D(128, (3, 3), padding="same")(up3)
up3 = concatenate([up3, conv2], axis=3)
deconv3 = Conv2D(128, (3, 3), padding="same")(up3)
deconv3 = Conv2D(128, (3, 3), padding="same")(deconv3)

# 第二层 上采样，卷积，通道拼接，卷积
up2 = UpSampling2D(size=(2,2))(deconv3)
up2 = Conv2D(64, (3, 3), padding="same")(up2)
up2 = concatenate([up2, conv1], axis=3)
deconv2 = Conv2D(64, (3, 3), padding="same")(up2)
deconv2 = Conv2D(64, (3, 3), padding="same")(deconv2)

# 第一层 卷积
output1 = Conv2D(3, (1, 1), padding="same")(deconv2)


# 第二部分，卷积部分
conv11 = Conv2D(64, (3, 3), padding="same")(input)
conv22 = Conv2D(128, (3, 3), padding="same")(conv11)
conv33 = Conv2D(64, (3, 3), padding="same")(conv22)
output2 = Conv2D(3, (3, 3), padding="same")(conv33)


# 两个输出点乘
output3 = keras.layers.Multiply()([output1, output2])


model = Model(input, output3)
model.compile(loss='mean_squared_error', optimizer='Nadam', metrics=['accuracy'])
model.summary()
