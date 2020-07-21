# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/21 002114:41
# 文件名称：孪生网络.py
# 开发工具：PyCharm

from keras.layers import *
import keras


# -----------------------------函数功能区-----------------------------
def FeatureNetwork():
    '''
    生成特征提取网络
    '''
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


# 孪生网络分为共享参数和不共享参数模型
def ClassifilerNet(reuse=True):  # reuse=True，便使用的是共享参数的模型
    """
    生成度量网络和决策网络，其实maychnet是两个网络结构，一个是特征提取（孪生）,一个是度量层+匹配层（统称为决策层）
    共享参数，一个model,训练参数为2,694,386
    不共享参数,两个model,参数为4,076,258
    """
    # 共享参数，一个model,训练参数为2,694,386
    if reuse:
        model = FeatureNetwork()
        inp1 = Input(shape=(28, 28, 1))  # 创建输入
        inp2 = Input(shape=(28, 28, 1))  # 创建输入2
        model_1 = model(inp1)  # 孪生网络中的一个特征提取分支
        model_2 = model(inp2)  # 孪生网络中的另一个特征提取分支
        merge_layers = concatenate([model_1, model_2])  # 进行融合，使用的是默认的sum，即简单的相加
    else: # 不共享参数,两个model,参数为4,076,258
        # ----------------------构造两个孪生网络--------------------------
        input1 = FeatureNetwork()  # 孪生网络中的一个特征提取
        input2 = FeatureNetwork()  # 孪生网络中的另一个特征提取
        # ----------------------构造两个孪生网络--------------------------

        # ----------------------对第二个孪生网络各层更名--------------------------
        for layer in input2.layers:
            layer.name = layer.name + str("_2")  # 网络重名会报错
        # ----------------------对第二个孪生网络各层更名--------------------------

        # ----------两个网络的输入数据
        inp1 = input1.input  # 第一个网络的输入
        inp2 = input2.input  # 第二个网络的输入

        # ----------两个网络层向量拼接
        # keras.layer中的concatenate
        merge_layers = concatenate([input1.output, input2.output])  # 进行融合，使用的是默认的sum，即简单的相加

    # ----------全连接
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    # ----------构建最终网络
    class_models = keras.Model([inp1, inp2], fc3)  # 最终网络架构，特征层+全连接层
    return class_models


ClassifilerNet().summary()