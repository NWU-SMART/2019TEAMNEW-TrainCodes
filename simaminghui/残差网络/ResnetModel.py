# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/26 002621:39
# 文件名称：ResnetModel.py
# 开发工具：PyCharm
from keras import Model
from keras.layers import *
from keras.regularizers import l2
class ResNet:

    # --------------------------------定义ResNet的残差模块residual_module-------------------------
    @staticmethod # 加上该注释表示此方法不用实例化也能调用
    def residual_module(x,K,stride,chanDim,reduce=False,reg=1e-4,bnEps=2e-5,bnMom=0.9):
        '''
        :param x: x为输入层
        :param K: K为最后输出的深度
        :param chanDim:定义将执行批归一化的轴
        :param reduce:是否降维
        :param bnEps:防止BN层出现除以0的异常
        :param bnMom:
        :return:
        '''
        shortcut = x

        # ----第一层卷积(归一、激活、1*1卷积)
        # 归一
        bn1 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
        # 激活
        act1 = Activation('relu')(bn1)
        # 1*1卷积
        conv1 = Conv2D(int(K*0.25),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1)

        # --------第二层卷积（归一，激活，1*1卷积）
        # 归一
        bn2 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
        act2 = Activation("relu")(bn2)
        conv2 = Conv2D(int(K*0.25),(3,3),strides=stride,padding="same",use_bias=False,kernel_regularizer=l2(reg))(act2)

        # -----------第三层卷积（归一，激活，1*1卷积）
        bn3 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv2)
        act3 = Activation("relu")(bn3)
        conv3 = Conv2D(K,(1,1),use_bias=False,kernel_regularizer=l2(reg))(act3)


        if reduce: # 是否降维，如果降维，需要将stride设置为大于1，更改shortcut的值
            shortcut = Conv2D(K,(1,1),strides=stride,kernel_regularizer=l2(reg))(act1)

        x = add([conv3,shortcut])  #该函数仅仅将shortcut部分和非shortcut部分相加在一起

        return x
    # --------------------------------定义ResNet的残差模块residual_module-------------------------

    # ------------------ 定义ResNet的网络构建模块build -------------
    '''
    width,height,depth为输入的shape。
    classes：表示最后分类的个数。如10分类，classes=10。
    
    '''
    @staticmethod
    def build(width,height,depth,classess,stages,filters,reg=1e-4,bnEps=2e-5,bnMom=0.9,dataset="cifar"):
        # --------初始化输入的高、宽和深度
        inputShape = (height,width,depth)
        chanDim = -1

        # ---设置输出
        input = Input(shape=inputShape)
        # ----归一化在这里第一层使用BN层而不是使用conv,这样可以替代取平均值的操作
        x = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(input)

        # 是否在cifar数据集上训练
        if dataset =="cifar":
            x = Conv2D(filters[0],(3,3),use_bias=False,padding="same",kernel_regularizer=l2(reg))(x)


        # -----步骤
        # -----1、按照层次数构建残差网络
        # -----2、构建全连接网络，进行分类


        # -----1、按照层次数构建残差网络
        # 构建多层残差网络
        # stages = [3,4,6]-->3(len(stages))个block，每个block中分别有3,4，6个残差块
        for i in range(0,len(stages)):
            if i ==0:
                stride = (1,1) # 如果是第一个block，则设置步长为（1，1），否则为（2,2）
            else:
                stride = (2,2)

            x = ResNet.residual_module(x,filters[i+1],stride=stride,chanDim=chanDim,reduce=True,bnEps=bnEps,bnMom=bnMom)  # 进行降维

            for j in range(0,stages[i]-1):
                x = ResNet.residual_module(x,filters[i+1],stride=(1,1),chanDim=chanDim,bnEps=bnEps,bnMom=bnMom)



        # 2、构建全连接网络，进行分类
        # 在构建深度残差网络后，构建分类器
        # 归一化
        x = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D((8,8))(x)
        x = Flatten()(x)
        x = Dense(classess,kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)

        model = Model(input,x,name="ResNet")

        model.summary()
        return model



