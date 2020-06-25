#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月24日
# 开发框架：Keras
# 开发内容：孪生网络（共享参数）
#----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 keras相关包
# 2、生成特征提取网络
# 3、生成度量网络
# 4、主函数
#--------------------------------------------------------------#

#  -------------------------- 1、导入需要包 -------------------------------
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.layers import Input
from keras.models import Model
#  -------------------------- 导入需要包 -------------------------------

# ---------------------2、生成特征提取网络-------------------------
def FeatureNetwork():
    """生成特征提取网络"""
    """这是根据，MNIST数据调整的网络结构，下面注释掉的部分是，原始的Matchnet网络中feature network结构"""

    # --- 输入数据
    inp = Input(shape = (28, 28, 1), name='FeatureNet_ImageInput')

    # ------------------------------ 网络第一层 --------------------------------------
    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)
    # ------------------------------ 网络第一层 --------------------------------------

    # ------------------------------ 网络第二层 --------------------------------------
    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
    models = Activation('relu')(models)
    # ------------------------------ 网络第二层 --------------------------------------

    # ------------------------------ 网络第三层 --------------------------------------
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    # ------------------------------ 网络第三层 --------------------------------------

    # ------------------------------ 网络第四层 --------------------------------------
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    # ------------------------------ 网络第四层 --------------------------------------

    # ------------------------------ 网络第五层（全连接） --------------------------------------
    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    model = Model(inputs=inp, outputs=models)
    # ------------------------------ 网络第五层（全连接） -------------------------------------

    return model
# ---------------------生成特征提取网络-------------------------

# ---------------------3、生成度量网络-------------------------
def ClassiFilerNet(reuse = True):

    """
    当reuse=True为孪生网络（共享参数）
    当reuse=False为伪孪生网络（不共享参数）
    生成度量网络和决策网络，其实maychnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)
    """
    # reuse = True:孪生网络
    if reuse:
        # --- 输入数据
        inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')
        # ------------------------------ 网络第一层 --------------------------------------
        models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
        models = Activation('relu')(models)
        models = MaxPool2D(pool_size=(3, 3))(models)
        # ------------------------------ 网络第一层 --------------------------------------

        # ------------------------------ 网络第二层 --------------------------------------
        models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
        # models = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(models)
        models = Activation('relu')(models)
        # ------------------------------ 网络第二层 --------------------------------------

        # ------------------------------ 网络第三层 --------------------------------------
        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        models = Activation('relu')(models)
        # ------------------------------ 网络第三层 --------------------------------------

        # ------------------------------ 网络第四层 --------------------------------------
        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        models = Activation('relu')(models)
        # ------------------------------ 网络第四层 --------------------------------------

        # ------------------------------ 网络第五层（全连接） --------------------------------------
        models = Flatten()(models)
        models = Dense(512)(models)
        models = Activation('relu')(models)
        model = Model(inputs=inp, outputs=models)
        # ------------------------------ 网络第五层（全连接） -------------------------------------
        inp1 = Input(shape = (28, 28, 1), name='FeatureNet_ImageInput1')
        inp2 = Input(shape = (28, 28, 1), name='FeatureNet_ImageInput2')
        model_1 = model(inp1)
        model_2 = model(inp2)
        merge_layers = concatenate([model_1, model_2])  # 进行融合，使用的是默认的sum，即简单的相加

    # reuse = False:伪孪生网络
    else:
        # ------------------- 构造两个孪生网络 --------------------
        input1 = FeatureNetwork()  # 孪生网络中的一个特征提取
        input2 = FeatureNetwork()  # 孪生网络中的另一个特征提取
        # ------------------- 构造两个孪生网络 --------------------

        # ------------------- 对于第二个网络各层更名 --------------------
        for layer in input2.layers:  # 这个for循环一定要加，否则网络重名会出错。
            layer.name = layer.name + str("_2")
        # ------------------- 对于第二个网络各层更名 --------------------

        # ----- 两个网络的输入数据
        inp1 = input1.input
        inp2 = input2.input

        # ----- 两个网络层向量拼接
        merge_layers = concatenate([input1.output, input2.output])  # 进行融合，使用的是默认的sum，即简单的相加

    # ----- 全连接
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    # ----- 构建最终网络
    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])

    return class_models
# ---------------------3、生成度量网络-------------------------

# ---------------------4、主程序-------------------------

matchnet = ClassiFilerNet()
matchnet.summary()  # 打印网络结构

# ---------------------主程序-------------------------