from keras import Input, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.saving import load_model
from keras.initializers import glorot_uniform
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Add, \
    Flatten, Dropout, Dense
from keras import layers
from DataProcess import *
from CBAM import *
import numpy as np
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


#ResNet50主要包括convolutional_block与identity_block两块

#定义identity_block
def identity_block(X, f, filters, stage, block):
    conv_name_base = "res" + str(stage) + block + "_branch"  #为每一卷积层命名
    bn_name_base = "bn" + str(stage) + block + "_branch"       #为每一BatchNormalization命名

    # Retrieve Filters
    F1, F2, F3 = filters   #卷积核数量

    X_shortcut = X

    #命名为res(stage)_branch2a的卷积层
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)  #批量归一化
    X = Activation("relu")(X)  #relu激活函数

    # 命名为res(stage)_branch2b的卷积层
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)    #批量归一化
    X = Activation("relu")(X)    #relu激活函数

    # 命名为res(stage)_branch2c的卷积层
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)  #批量归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)   #relu激活函数

    X = Add()([X, X_shortcut])  #与上一层X_shortcut进行连接
    X = Activation("relu")(X)    #激活函数
    ### END CODE HERE ###

    return X


#定义convolutional_block
def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'  #为每一卷积层命名
    bn_name_base = 'bn' + str(stage) + block + '_branch'     #为每一BatchNormalization命名

    # Retrieve Filters
    F1, F2, F3 = filters   #每层卷积核数量

    # 保存输入值
    X_shortcut = X

    # 卷积、批量归一化、relu
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)


    # 卷积、批量归一化、relu
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # 卷积、批量归一化
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    #卷积、批量归一化、激活函数
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])  #X与X_shortcut连接
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X

inputs_all = Input(shape=(256, 256, 3))  #输入数据


# Zero-Padding
X = ZeroPadding2D((3, 3))(inputs_all)  #该图层可以在图像张量的顶部、底部、左侧和右侧添加零表示的行和列。

#第一部分，先做一个简单的卷积、批量归一化、relu、MaxPooling2D；第1层卷积
X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",
          kernel_initializer=glorot_uniform(seed=0))(X)  #64*64*64
X = BatchNormalization(axis=3, name="bn_conv1")(X)
X = Activation("relu")(X)
X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxPooling1')(X)    #32*32*64

#第二部分：先进行一个卷积块，再进行两个残差块；第2-10层卷积
X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)  #16*16*256
X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")            #16*16*256
X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")            #16*16*256

# 第三部分：先进行一个卷积快，再进行三个残差块；第11-22层卷积
X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=1) #8*8*512
X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")           #8*8*512
X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")           #8*8*512
X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")           #8*8*512

# 第四部分：先进行一个卷积块，再进行5个残差块；第23-40层卷积
X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2) #4*4*1024
X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")           #4*4*1024
X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")           #4*4*1024
X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")           #4*4*1024
X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")           #4*4*1024
X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")           #4*4*1024

# 第五部分：先进行一个卷积块，再进行2个残差块；第41-49层卷积
X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2) #2*2*2048
X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")           #2*2*2048
X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")           #2*2*2048

#平均池化
X = AveragePooling2D(pool_size=(2, 2), padding="same", name='averagePool1')(X)
X = cbam_block(X, 8)   #注意力机制
X = AveragePooling2D(pool_size=(2, 2), padding="same", name='averagePool2')(X)
#Global_feature = spatial  #得到全局特征
#print('ResNet50得到的全局特征:')
#print(Global_feature)
X = Flatten()(X)
X = Dropout(0.5)(X)
X = Activation('relu')(X)
X = Dropout(0.3)(X)
X = Activation('relu')(X)
out = Dense(567, activation='softmax', name='dense')(X)

model = Model(inputs=inputs_all, outputs=out)
adam = optimizers.Adam(lr=0.0000001,beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])  #模型训练的BP模式设置
model.summary()

filepath = "resnet.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=2, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=2)
callbacks_list = [checkpoint]

model.fit(XRay_train, YXTrain, epochs=100, batch_size=16, validation_split=0.1, callbacks=callbacks_list)
predict = model.evaluate(XRay_train, YXTrain, batch_size=16)
print("测试集得到的损失和测试精度：")
print("Loss = " + str(predict[0]))  # 测试损失
print("Test Accuracy = " + str(predict[1]))  # 测试精度

#model.save('resnet.h5')
#加载resnet模型
modelResnet = load_model('resnet.h5')
resnet_model = Model(inputs=modelResnet.input, outputs=modelResnet.get_layer('averagePool2').output)
resnet_model.summary()
#冻结模型的所有层
for layer in resnet_model.layers[:]:
    layer.trainable = False
#创建模型
modelAdd = Sequential()
modelAdd.add(resnet_model)   #加入resnet的基础卷积网络
modelAdd.add(Flatten())
modelAdd.add(Dropout(0.5))
modelAdd.add(Activation('relu'))
modelAdd.add(Dropout(0.3))
modelAdd.add(Activation('relu'))
modelAdd.add(Dense(2, activation='softmax', name='dense2'))
modelAdd.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
modelAdd.summary()
modelAdd.fit(X_train, Y_train, epochs=100, batch_size=16, validation_split=0.05 )
predict = modelAdd.evaluate(X_test, Y_test, batch_size=16)
print("测试集得到的损失和测试精度：")
print("Loss = " + str(predict[0]))  # 测试损失
print("Test Accuracy = " + str(predict[1]))  # 测试精度
modelAdd.save('finally.h5')


