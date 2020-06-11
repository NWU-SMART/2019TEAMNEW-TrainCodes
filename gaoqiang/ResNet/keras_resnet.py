# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.11
# 开发框架：keras
# 代码功能：ResNet
#----------------------------------------------------------------------------------------------------------------------#

'''
ResNet:
目前神经网络变得越来越复杂，从几层到几十层甚至一百多层的网络都有。深层网络的主要的优势是可以表达非常复杂的函数，它可以从不
同抽象程度的层里学到特征，比如在比较低层次里学习到边缘特征而在较高层里可以学到复杂的特征。然而使用深层网络并非总是奏效，
因为存在一个非常大的障碍——梯度消失：在非常深层的网络中，梯度信号通常会非常快的趋近于0，这使得梯度下降的过程慢得令人发
指。具体来说就是在梯度下降的过程中，从最后一层反向传播回第一层的每一步中都要进行权重矩阵乘积运算，这样一来梯度会呈指数级
下降到0值。（在极少的情况下还有梯度爆炸的问题，就是梯度在传播的过程中以指数级增长到溢出）
'''
# resnet做加和操作，因此用add函数，
# googlenet以及densenet做filter的拼接，因此用concatenate
# add和concatenate的区别参考链接：https://blog.csdn.net/u012193416/article/details/79479935

from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.regularizers import l2   # 这里加入l2正则化的目的是为了防止过拟合
from keras.layers import add,Input
import keras.backend as K
from keras.models import Model



class ResNet:
    #---------------------------定义ResNet的残差模块residual_module----------------------------------------------------#
    @staticmethod   # 静态方法 类或实例均可调用 # 改静态方法函数里不传入self 或 cls
    def residual_module(x,K,stride,chanDim,reduce =False,reg=1e-4,bnEps=2e-5,bnMom=0.9):
        '''
        Parameters:
            x: residual module的输入.
            K: 最终卷积层的输出通道
            stride: 步长
            chanDim: 定义批归一化处理的axis（坐标轴）
            reduce: 是否降维，
            reg: 正则化强度
            bnEps: 防止BN层出现除以0的异常
            bnMom: Controls the momentum for the moving average.
        Return:
            x: Return the output of the residual module.
        '''

        # 获取的结果为F(x)+x,这里输入的信号为x(直接跳转)，F(x)为经过多个卷积处理函数
        # F(x)经过的卷积操作为1x1（第一层），3x3（第二层）和1x1（第三层）
        shortcut = x  # ResNet模块的快捷分支应该初始化为输入(标识)数据。

        bn1 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
        act1 = Activation("relu")(bn1)
        # 因为偏差在紧跟着卷积的BN层中，所以没有必要引入#第二个* bias项，因此改变了典型的CONV块顺序
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)  # 输出通道 K*0.25,kernel_size=(1,1),stride=(1,1)

        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3),strides=stride,padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K ,(1, 1),use_bias=False, kernel_regularizer=l2(reg))(act3)

        # 如果我们想降维，应用CONV层的快捷分支
        if reduce: #是否降维，如果降维的话，需要将stride设置为大于1,更改shortcut值
            shortcut = Conv2D(K ,(1, 1),strides=stride,use_bias=False, kernel_regularizer=l2(reg))(act1)

        # 这个与googlenet的concatenate函数不同，add函数做简单加法，concatenate函数做横向拼接.该函数仅仅将shortcut部分和
        # 非shortcut部分相加在一起
        x = add([conv3,shortcut])  # 计算F(x)+x

        return x   # 输出结果= conv3+shortcut


    #-----------------------------------定义ResNet的网络构建模块build -------------------------------------------------#

    @staticmethod  # 静态方法 类或实例均可调用 # 改静态方法函数里不传入self 或 cls
    def build(width,height,depth,classes,stages,filters,reg=1e-4,bnEps=2e-5,bnMom=0.9,dataset="cifar"):
        # 初始化输入shape，变为"channels last"，初始化通道大小
        inputShape = (height,width,depth)
        chanDim = -1

        # 如果是"channels first"，初始化通道shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth,height,width)
            chanDim = 1

        input =Input(shape = inputShape)
        # 在这里第一层使用BN层而不是使用conv，这样可以替代取平均值的操作(省去对于图像的均值处理)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(input)

        # 是否在CIFAR数据集上训练
        if dataset == "cifar":
            # 应用第一个卷积层
            x = Conv2D(filters[0],(3,3),use_bias=False,padding="same",kernel_regularizer=l2(reg))(x)
        '''
        # ----- 步骤
        # ----- 1、按照层次数构建残差网络（多少个block，每个block中有多少个残差块）
        # ----- 2、构建全连接网络，进行分类

        # 构建多层残差网络（len(stages)）
        # stages=[3, 4, 6] --> 3个block，每个block中分别有3,4,6个残差块
        
        '''
        for i in range(0,len(stages)): # 每阶段的遍历
            stride =(1,1) if i == 0 else (2,2)  # 第一层步长设置为1，其余层为2
            x = ResNet.residual_module(x,filters[i+1],stride=stride,chanDim=chanDim,reduce =True,bnEps=bnEps,bnMom=bnMom)# 降维
            for j in range(0,stages[i]-1): # 每层的遍历
                x = ResNet.residual_module(x,filters[i+1],stride=(1,1),chanDim=chanDim,bnEps=bnEps,bnMom=bnMom)# 不降维

        # 在将所有剩余模块堆叠在一起之后，即构建深度残差网络后，构建分类器
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8,8))(x)

        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes,kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        # 构建模型
        model = Model(input,x,name="ResNet")
        # 打印网络结构
        model.summary()

        return model














