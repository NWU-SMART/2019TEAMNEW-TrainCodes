# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月29日
# 内容:Siamese Network(共享参数）
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、函数功能区
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model
import keras
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、函数功能区 -----------------------------------
'''
利用keras实现孪生网络中的权值共享，权值不是cnn中的共享权值，而是如何在构建类似于Sianese Network这样的多分枝网络，且分支结构相同时，如何使用keras
使分支的权重共享
Siamese network就是看两张图片的相似性，是一种多输入单输出的网络结构，输入是两张图片，两个向量，输出为是否相似，多输入单输出的分类问题
o:不相似，1：相似
这个是共享参数的模型，是以MatchNet网路结构，为方便显示，将卷积模块减为2个
'''
#from keras.datasets import mnist#导入手写体数据集
#特征提取
def FeatureNetwork():
    inp = Input(shape=(28,28,1),name='FeatureNet_ImageInput')#通过一个'name'参数来命名任何层
     ##提取特征网路第一层
    models=Conv2D(filters=24,kernel_size=(3,3),strides=1,padding='same')(inp)
    models=Activation('relu')(models)
    models=MaxPool2D(pool_size=(3,3))(models)
    ##提取特征网路第二层
    models=Conv2D(filters=64,kernel_size=(3,3),strides=1,padding='same')(models)
    models=Activation('relu')(models)
    ##提取特征网路第三层
    models=Conv2D(filters=96,kernel_size=(3,3),strides=1,padding='valid')(models)
    models=Activation('relu')(models)
    ##提取特征网路第四层
    models=Conv2D(filters=96,kernel_size=(3,3),strides=1,padding='valid')(models)
    moedls=Activation('relu')(models)
    ##提取特征网路第五层
    models=Flatten()(models)
    models=Dense(512)(models)
    models=Activation('relu')(models)
    model=Model(inputs=inp,outputs=models)
    return model
#分类网络， """生成度量网络和决策网络，其实maychnet是两个网络结构，一个是特征提取层(孪生)，一个度量层+匹配层(统称为决策层)"""
def ClassiFilerNet(reuse = True):
    if reuse:         #如果参数共享
        inp=Input(shape=(28,28,1),name='FeatureNet_ImageInput')
        models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
        models = Activation('relu')(models)
        models = MaxPool2D(pool_size=(3, 3))(models)
        ##提取特征网路第二层
        models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
        models = Activation('relu')(models)
        ##提取特征网路第三层
        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        models = Activation('relu')(models)
        ##提取特征网路第四层
        models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
        moedls = Activation('relu')(models)
        ##提取特征网路第五层
        models = Flatten()(models)
        models = Dense(512)(models)
        models = Activation('relu')(models)
        model=Model(inputs=inp,outputs=models)
        inp1=Input(shape=(28,28,1))#创建输入
        inp2=Input(shape=(28,28,1))  # 创建输入2
        model_1=model(inp1)#孪生网路中的一个特征提取分支
        model_2=model(inp2)#孪生网路中的另一个特征提取分支
        merge_layers=concatenate([model_1,model_2])#进行融和,使用简单相加
    else:
        input1 = FeatureNetwork()  # 调用了两次FeatureNetwork，所以生成的input1和input2是两个完全独立的模型分支，参数不共享
        input2 = FeatureNetwork()
        # 对第二个网络各层更改名字
        for layer in input2.layers:
            layer.name = layer.name + str("_2")
        inp1 = input1.input  # 两个网络的输入数据
        inp2 = input2.input  # 两个网络的输入数据
        # 两个网络层的向量进行merge,使用的是concatenate,简单加法
        merge_layers = concatenate([input1.output, input2.output])
        # 进行分类运算，三个全连接层
    fc1 = Dense(1024, activation='relu')(merge_layers)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)
        # 构建最终网路
    class_models = Model(inputs=[inp1, inp2], outputs=[fc3])
    return class_models
#  ---------------------- 2、函数功能区 -----------------------------------
matchnet = ClassiFilerNet()
matchnet.summary() # 打印网络结构
plot_model(matchnet,to_file='D:/northwest/小组视频model.png')
