#--------------------------------------------------------开发者信息------------------------------------------------------
#开发者：王园园
#开发日期：2020.6.04
#开发软件：pycharm
#开发项目：孪生网络（keras）

#-------------------------------------------------------------导包-------------------------------------------------------
from keras import Input, Model
from keras.layers import Conv2D, Activation, MaxPool2D, Flatten, Dense, concatenate
from keras.utils import plot_model

#-------------------------------------------------------------函数功能区-------------------------------------------------
def FeatureNetwork():
    #API方式构建模型
    # 输入数据
    inp = Input(shape=(28, 28, 1), name='FeatureNet_ImageInput')
    #网络第一层
    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)
    #网络第二层
    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    models = Activation('relu')(models)
    #网络第三层
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    #网络第四层
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)
    #网络第五层：全连接
    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    model = Model(inputs=inp, outputs=models)

    return  model

def ClassiFilerNet():
    #生成度量网络和决策网络，其实matchnet是两个网络结构，一个是特征提取层（孪生），一个度量层+匹配层（统称为决策层）
    #构造两个孪生网络:都是特征提取
    input1 = FeatureNetwork()
    input2 = FeatureNetwork()

    #对于第二个网络各层更改名字
    for layer in input2.layers:
        layer.name = layer.name + str('_2')

    #两个网络的输入数据
    inp1 = input1.input
    inp2 = input2.input

    #两个网络层向量拼接:向量进行融合，使用的是默认的sum，既简单的相加
    merge_layers = concatenate([input1.output, input2.output])

    #全连接
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    #构建最终网络
    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])

    return class_models

#------------------------------------------------------------------主调区------------------------------------------------
matchnet = ClassiFilerNet()
matchnet.summary()
plot_model(matchnet, to_file='D:\images')  #网络结构输出成png图片