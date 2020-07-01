# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.29
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# ----------------------   代码布局： ----------------------
# 1、导入 Keras的包
# 2、定义残差网络ResNet的类
# ----------------------   代码布局： ----------------------


#  -------------------------- 1、导入需要包 -------------------------------
from keras.layers import BatchNormalization,Conv2D,\
    AveragePooling2D,MaxPool2D,ZeroPadding2D,Activation,Flatten,Dense,Dropout,Input,add
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
from keras.utils.vis_utils import plot_model

#  -------------------------- 1、导入需要包 -------------------------------

class Resnet:
    @staticmethod
    def residual_module(x,K,stride,chanDim,reduce=False,reg=1e-4,bnEps=2e-5,bnMom=0.9):
        '''
        :param x: 残差块的输入
        :param K:最终卷积的输出
        :param stride:步长
        :param chanDim:批量归一化的轴
        :param reduce:判断是否降维
        :param reg:正则化系数
        :param bnEps:防止出现Bn层为0
        :param bnMom:平均移动量
        :return:返回剩余模块的输出
        '''
        # 确定输入为x，并且保证残差的shorcut同样为x
        shortcut = x
        bn1 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
        act1 = Activation('relu')(bn1)
        # 输出维度int(K*0.25)
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)
        '''
        axis: 整数，需要标准化的轴 （通常是特征轴）。 
        例如，在 data_format="channels_first" 的 Conv2D 层之后， 在 BatchNormalization 中设置 axis=1。
        momentum: 移动均值和移动方差的动量。
        epsilon: 增加到方差的小的浮点数，以避免除以零。
        '''
        bn2 = BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(conv1)
        act2 = Activation('relu')(bn2)
        # 输出维度int(K*0.25)
        conv2 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act2)

        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation('relu')(bn3)
        # 输出维度int(K*0.25)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # 判断是否需要降维处理,通过stride的值实现降维
        if reduce:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)


        # 将正常的处理与残差处理的结合到一块
        x = add([conv3,shortcut])
        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset="cifar"):

        #初始化图片的高、宽和深度
        inputShape = (height, width, depth)
        # 将通道变成最后一维
        chanDim = -1
        # 将顺序变成channels_last
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        #设置输入大小
        input = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(input)
        # 是否在CIFAR数据集上训练
        if dataset == "cifar":
            # 调用卷积层进行处理
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

        # 需要调用的各个网络已经定义完毕

        # ----- 1、按照层次数构建残差网络（多少个block，每个block中有多少个残差块）
        # ----- 2、构建全连接网络，进行分类

        for i in range(0,len(stages)):
            stride = (1,1) if i==0 else (2,2)
            # 降维
            x = Resnet.residual_module(x,filters[i+1],stride=stride,chanDim=chanDim,reduce =True,bnEps=bnEps,bnMom=bnMom)
            for j in range(0,stages[i]-1): # 每层的遍历
                # 不降维
                x = Resnet.residual_module(x,filters[i+1],stride=(1,1),chanDim=chanDim,bnEps=bnEps,bnMom=bnMom)
            # 对x进行继续处理，构建一个分类器
            x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
            x = Activation("relu")(x)
            x = AveragePooling2D((8, 8))(x)


            x = Flatten()(x)
            x = Dense(classes, kernel_regularizer=l2(reg))(x)
            x = Activation("softmax")(x)

            # 构建模型
            model = Model(input, x, name="ResNet")
            # 打印网络结构

            model.summary()



            return model


# 构建残差网络
# 进行了三次残差网络
model = Resnet.build(32, 32, 3, 10, stages=[3, 4, 6], filters=[64, 128, 256, 512])  # 因为googleNet默认输入32*32的图片
# 输出模型
plot_model(model, to_file="resnet.png", show_shapes=True)