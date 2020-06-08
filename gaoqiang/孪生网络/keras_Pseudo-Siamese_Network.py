# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.08
# 开发框架：keras
# 温馨提示：伪孪生网络
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局------------------------------------------------------------#
# 1、建立模型
# 2、保存模型与模型可视化
#----------------------------------------------------------------------------------------------------------------------#
'''
任务介绍：孪生神经网络(Siamese Network)
问题的提出与解决方案：
分类问题：
第一类，分类数量较少，每一类的数据量较多，比如ImageNet、VOC等。这种分类问题可以使用神经网络或者SVM解决，只要事先知道了
所有的类。
第二类，分类数量较多（或者说无法确认具体数量），每一类的数据量较少，比如人脸识别、人脸验证任务。
文中提出的解决方案：
learn a similar metric from data。核心思想是，寻找一个映射函数，能够将输入图像转换到一个特征空间，每幅图像对应一个特征
向量，通过一些简单的“距离度量”（比如欧式距离）来表示向量之间的差异，最后通过这个距离来拟合输入图像的相似度差异（
语义差异）。

简单来说，衡量两个输入的相似程度。孪生神经网络有两个输入（Input1 and Input2）,将两个输入feed进入两个神经网络
（Network1 and Network2），这两个神经网络分别将输入映射到新的空间，形成输入在新的空间中的表示。通过Loss的计算，评价两个
输入的相似度。

之所以称之为孪生网络，是因为“权值共享”，如果不共享权值，则称之为 pseudo-siamese network，伪孪生神经网络。
孪生神经网络用于处理两个输入"比较类似"的情况。伪孪生神经网络适用于处理两个输入"有一定差别"的情况。
'''
#-------------------------------搭建伪孪生神经网络模型（不共享参数）-------------------------------------------------#
from keras.layers import Input,Conv2D,Activation,MaxPool2D,Flatten,Dense,concatenate
from keras.models import Model

def FeatureNetwork(): # 生成特征提取网络
     input = Input(shape = (28,28,1),name='FeatureNet_ImageInput')
     x = Conv2D(filters=24,kernel_size=(3,3),strides=1,padding='same')(input) # 输出24通道
     x = Activation('relu')(x)
     x = MaxPool2D(pool_size=(3,3))(x)

     x = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(x)
     x = Activation('relu')(x)

     x = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(x) #  padding='valid' 即不加padding
     x = Activation('relu')(x)

     x = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(x)  # padding='valid' 即不加padding
     x = Activation('relu')(x)
     x = Flatten()(x)
     x = Dense(512)(x)
     x = Activation('relu')(x)

     model = Model(inputs=input,outputs=x)
     return model

def ClassiFilerNet():                  # 生成度量网络和决策网络
    input1 = FeatureNetwork()          # 孪生网络的一个特征提取
    input2 = FeatureNetwork()          # 孪生网络的另一个特征提取

    for layer in input2.layers:        # 为了避免重名
        layer.name = layer.name + str("_2")
    inp1 = input1.input
    inp2 = input2.input
    merge_layers = concatenate([input1.output,input2.output])# 进行融合，使用的是默认的sum，即简单的相加
    fc1 = Dense(1024,activation='relu')(merge_layers)
    fc2 = Dense(1024, activation='relu')(fc1)
    fc3 = Dense(2, activation='softmax')(fc2)

    class_models = Model(inputs=[inp1,inp2],outputs=[fc3])
    return class_models

# 主调区
matchnet = ClassiFilerNet()
matchnet.summary() # 打印网络结构
from keras.utils.vis_utils import plot_model
plot_model(matchnet,to_file='pseudo-siamese network model.png')
#----------------------------------------------------------------------------------------------------------------------#



















#----------------------------------------------------------------------------------------------------------------------#