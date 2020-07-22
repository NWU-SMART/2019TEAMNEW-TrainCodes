# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/22 002217:43
# 文件名称：4代码实现
# 开发工具：PyCharm

import numpy as np
import keras
from keras.layers import *

path = 'siameseData.npz'
f = np.load(path)

x1, x2, Y = f["x1"], f['x2'], f['Y']
x1 = x1.astype('float32')/255.
x2 = x2.astype('float32')/255.
x1 = x1.reshape(60000,28,28,1)
print(x2.shape)
x2 = x2.reshape(60000,28,28,1)
print(Y.shape)



# ---------------------查看相同数字（不同数字）个数----------------------
def sum():
    oneSum = 0
    zerosSum = 0
    for i in range(60000):
        if Y[i] == 1:
            oneSum = oneSum + 1
        else:
            zerosSum = zerosSum + 1
    print("相同的个数{}".format(oneSum))
    print("不同的个数{}".format(zerosSum))


sum()  # 相同的个数30000，不同的个数30000


# ---------------------查看相同数字（不同数字）个数----------------------


# -----------------------开始孪生网络构建--------------------------------------

# 特征提取，对两张图片进行特征提取
def FeatureNetwork():
    F_input = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')

    # ----------------------------------网络第一层----------------------
    # 28,28,1-->28,28,24
    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(F_input)
    models = Activation('relu')(models)
    # 28,28,24-->9,9,24
    models = MaxPooling2D(pool_size=(3, 3))(models)
    # ----------------------------------网络第一层----------------------

    # ----------------------------------网络第二层----------------------
    # 9,9,24-->9,9,64
    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    models = Activation('relu')(models)
    # ----------------------------------网络第二层----------------------

    # ----------------------------------网络第三层----------------------
    # 9,9,64-->7,7,96
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    # ----------------------------------网络第三层----------------------

    # ----------------------------------网络第四层----------------------
    # 7,7,96-->5,5,96
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    # ----------------------------------网络第四层----------------------

    # ----------------------------------网络第五层----------------------
    # 5,5,96-->2400
    models = Flatten()(models)
    # 2400-->512
    models = Dense(512)(models)
    models = Activation('relu')(models)
    # ----------------------------------网络第五层----------------------

    return keras.Model(F_input, models)


# 共享参数
def ClassifilerNet():
    model = FeatureNetwork()
    inp1 = Input(shape=(28, 28, 1))  # 创建输入
    inp2 = Input(shape=(28, 28, 1))  # 创建输入2
    model_1 = model(inp1)  # 孪生网络中的一个特征提取分支
    model_2 = model(inp2)  # 孪生网络中的另一个特征提取分支
    merge_layers = concatenate([model_1, model_2])  # 进行融合，使用的是默认的sum，即简单的相加
    # ----------全连接---------
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(1, activation='sigmoid')(fc2)

    # ----------构建最终网络
    class_models = keras.Model([inp1, inp2], fc3)  # 最终网络架构，特征层+全连接层
    return class_models


#-----------------------孪生网络实例化以及编译训练-----------------------
siamese_model = ClassifilerNet()
siamese_model.summary()


siamese_model.compile(loss='mse', # 损失函数采用mse
                      optimizer='rmsprop',
                      metrics=['accuracy']
                      )



history = siamese_model.fit([x1,x2],Y,
                            batch_size=256,
                            epochs=2,
                            validation_split=0.2)


#-----------------------孪生网络实例化以及编译训练end-----------------------
siamese_model.save('siamese_model2.h5')
print(history.history.keys())

# ----------------------查看效果-------------------------------
import matplotlib.pyplot as plt
# 准确
plt.plot(history.history['accuracy']) # 训练集准确率
plt.plot(history.history['val_accuracy']) # 验证集准确率
plt.legend()
plt.show()
# 画损失
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend()
plt.show()
# ----------------------查看效果end-------------------------------



#---------------------后面测试（test）最后代码----------------------------

#---------------------后面测试（test）最后代码end----------------------------
# -----------------------开始孪生网络构建end--------------------------------------
