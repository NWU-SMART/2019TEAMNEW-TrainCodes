#-------------------------------------------------------开发者信息--------------------------------------------------------
#开发者：王园园
#开发日期：2020.3 自编码器
#开发软件：pycharm
#开发项目：残差网络模型

#----------------------------------------------------------导包-------------------------------------------------------=-
from IPython.core.magics.config import reg
from keras import Input, Model
from keras.layers import Activation, Conv2D, add, AveragePooling2D, Flatten, Dense
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.regularizers import l2
from keras.utils import plot_model

#---------------------------------------------------------定义残差网络ResNet的类-----------------------------------------
class ResNet:
    # 定义ResNet的残差模块residual_module
    @staticmethod
    def residual_module(x, k, stride, chanDim, reduce=False, reg=1e-4, bnEps=2e-5, bnMom=0.9):
        #获取的结果为F（x）+x， 这里输入的信号为x（直接跳转）， F（x）为经过多个卷积处理函数
        #F（x）经过的卷积操作为1*1（第一层）， 3*3（第二层）和1*1（第三层）
        shortcut = x
        #第一层卷积（归一、激活和1*1卷积）
        #归一
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)
        #激活
        act1 = Activation('relu')(bn1)
        #1*1卷积
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)

        #第二层卷积
        #归一
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides = stride, padding='same', use_bias=False, kernel_regularizer=l2(reg))(act2)

        #第三层卷积
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)
        #是否降维，如果降维的话，需要将stride设置为大于1， 更改shortcut的值
        if reduce:
            shortcut = Conv2D(K, (1, 1), strides = stride, use_bias=False, kernel_regularizer=l2(reg))(act1)
        #计算F(x)+x
        #这个与googlenet的concatenate函数不同， add函数做简单加法， concatenate函数做横向拼接， 该函数仅仅将shortcut部分和非shortcut部分相加在一起
        x = add([conv3, shortcut])
        #函数返回F(x)+x
        return x

# -----------------------------------------------------------定义ResNet的网络结构模块build---------------------------------
    @staticmethod
    def build(width, height, depth, classes, stages, filters, red=1e-4, bnEps=2e-5, bnMom=0.9, dataset='cifer'):
        # 初始化输入的高、宽和深度
        inputShape = (height, width, depth)
        chanDim = -1

        #如果顺序为‘channels first’， 更改顺序
        if K.image_data_format() == 'channels_first':
            inpuktShape = (depth, height, width)
            chanDim = 1

        #设置输入
        input = Input(shaep=inputShape)
        #归一化在这里第一层使用BN层而不是使用conv， 这样可以替代取平均值的操作
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(input)

        #是否在CIFAR数据集上训练
        if dataset == 'cifar':
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)
        #步骤
        #1、按照层次数构建残差网络（多少个blok, meige1block中有多少个残差块）
        #2、构建全连接网络，进行分类
        #构建多层残差网络(len(stages))
        #stages=[3,4,6]-->3个block，每个block中分别有3，4，6个残差块
        for i in range(0, len(stages)):
            # 构建第一个block，设置步长为（1， 1）， 否则为（2， 2）
            stride = (1, 1) if i == 0 else (2, 2)
            #降维
            x = ResNet.residual_module(x, filters[i+1], stride=(1, 1), chanDim=chanDim, reduce=True, bnEps=bnEps, bnMom = bnMom)
            for j in range(0, stages[i]-1):
                #不进行降维
                x = ResNet.residual_module(x, filters[i+1], stride=(1, 1), chanDim=chanDim, bnEps=bnEps, bnMom=bnMom)

        #在构建深度残差网络后，构建分类器
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)

        model = Model(input, x, name='ResNet')
        model.summary()
        return model

#--------------------------------------------------------模型调用--------------------------------------------------------
model = ResNet.build(32, 32, 3, 10, stages=[3, 4, 6], filters=[64, 128, 256, 512])
plot_model(model, to_file='output/resnet.png', show_shapes=True)











