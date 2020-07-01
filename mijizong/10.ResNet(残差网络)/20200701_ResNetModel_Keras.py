# ----------------------开发者信息-----------------------------------------
#  -*- coding: utf-8 -*-
#  @Time: 2020/7/1
#  @Author: MiJizong
#  @Content: ResNet——Keras
#  @Version: 1.0
#  @FileName: 1.0.py
#  @Software: PyCharm
#  @Remarks: Null
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： -------------------------------------
# 1、导入 Keras的包
# 2、定义残差网络ResNet的类
#      定义ResNet的残差模块residual_module
#      定义ResNet的网络构建模块build
# 3、模型调用
# 第三步也可以单独写程序调用
# ----------------------   代码布局： -------------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import Input, add
from keras.models import Model
from keras.regularizers import l2  # 这里加入l2正则化目的是为了防止过拟合
from keras.utils.vis_utils import plot_model
import keras.backend as K


#  -------------------------- 1、导入需要包 -------------------------------


# resnet做加和操作，因此用add函数，
# googlenet以及densenet做filter的拼接，因此用concatenate
# add和concatenate的区别参考链接：https://blog.csdn.net/u012193416/article/details/79479935

#  -------------------------- 2、定义残差网络ResNet的类 -------------------------------
class ResNet:

    # ------------------ 定义ResNet的残差模块residual_module -------------
    @staticmethod
    def residual_module(x, K, stride, chanDim, reduce=False, reg=1e-4, bnEps=2e-5, bnMom=0.9):
        """
        The residual module of the ResNet architecture.
        Parameters:
            x: The input to the residual module.
            K: The number of the filters that will be learned by the final CONV in the bottlenecks.最终卷积层的输出.
            stride: Controls the stride of the convolution, help reduce the spatial dimensions of the volume *without*
                resorting to max-pooling.
            chanDim: Define the axis which will perform batch normalization.定义将执行批量归一化的轴。
            reduce: Cause not all residual module will be responsible for reducing the dimensions of spatial volums -- the
                red boolean will control whether reducing spatial dimensions (True) or not (False).是否降维，
            reg: Controls the regularization strength to all CONV layers in the residual module.控制正则化强度
            bnEps: Controls the ε responsible for avoiding 'division by zero' errors when normalizing inputs.防止BN层出现除以0的异常
            bnMom: Controls the momentum for the moving average.
        Return:
            x: Return the output of the residual module.
        """

        # 获取的结果为F(x)+x,这里输入的信号为x(直接跳转)，F(x)为经过多个卷积处理的函数
        # F(x)经过的卷积操作为1x1（第一层），3x3（第二层）和1x1（第三层）
        shortcut = x

        # ------ 第一层卷积(归一、激活和1x1卷积)
        # 归一
        bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        # 激活
        act1 = Activation("relu")(bn1)
        # 1x1卷积
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)
        # filter=K*0.25,kernel_size=(1,1),stride=(1,1)

        # ------ 第二层卷积(归一、激活和1x1卷积)
        # 归一
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        # 激活
        act2 = Activation("relu")(bn2)
        # 3x3卷积
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                       kernel_regularizer=l2(reg))(act2)

        # ------ 第三层卷积(归一、激活和1x1卷积)
        # 归一
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        # 激活
        act3 = Activation("relu")(bn3)
        # 1x1卷积
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if reduce:  # 是否降维，如果降维的话，需要将stride设置为大于1,更改shortcut值
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # 计算F(x)+x
        x = add([conv3, shortcut])
        # 这个与googlenet的concatenate函数不同，add函数做简单加法，concatenate函数做横向拼接.该函数仅仅将shortcut部分和非shortcut部分相加在一起

        # 函数返回F(x)+x
        return x  # f(x)输出结果 = conv3+shortcut

    # ------------------ 定义ResNet的残差模块residual_module -------------

    # ------------------ 定义ResNet的网络构建模块build -------------
    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=1e-4, bnEps=2e-5, bnMom=0.9, dataset="cifar"):

        # --- 初始化输入的高、宽和深度
        inputShape = (height, width, depth)
        chanDim = -1

        # --- 如果顺序为 "channels first", 更改顺序
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # ---- 设置输出
        input = Input(shape=inputShape)
        # --- 归一化在这里第一层使用BN层而不是使用conv，这样可以替代取平均值的操作
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(input)

        # 是否在CIFAR数据集上训练
        if dataset == "cifar":
            # Apply the first and single CONV layer.
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)

        # ----- 步骤
        # ----- 1、按照层次数构建残差网络（多少个block，每个block中有多少个残差块）
        # ----- 2、构建全连接网络，进行分类

        # ----- 1、按照层次数构建残差网络（多少个block，每个block中有多少个残差块）
        # 构建多层残差网络（len(stages)）
        # stages=[3, 4, 6] --> 3个block，每个block中分别有3,4,6个残差块
        for i in range(0, len(stages)):  # 每阶段的遍历

            # --- 初始步长，
            # 构建第一个block，在第一项中设置步长为 (1,1)，否则为(2,2)
            stride = (1, 1) if i == 0 else (2, 2)

            # 进行降维
            x = ResNet.residual_module(x, filters[i + 1], stride=stride, chanDim=chanDim, reduce=True, bnEps=bnEps,
                                       bnMom=bnMom)

            # 循环遍历
            for j in range(0, stages[i] - 1):  # 每层的遍历
                # 不进行降维
                x = ResNet.residual_module(x, filters[i + 1], stride=(1, 1), chanDim=chanDim, bnEps=bnEps,
                                           bnMom=bnMom)
        # ----- 1、按照层次数构建残差网络（多少个block，每个block中有多少个残差块）

        # ----- 2、构建全连接网络，进行分类
        # 在构建深度残差网络后，构建分类器
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)  # 归一化
        x = Activation("relu")(x)  # 激活
        x = AveragePooling2D((8, 8))(x)  # 平均池化
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)  # Softmax classifier.
        x = Activation("softmax")(x)
        # ----- 2、构建全连接网络，进行分类

        # 构建模型
        model = Model(input, x, name="ResNet")

        model.summary()  # 输出网络结构信息
        return model
    # ------------------ 定义ResNet的网络构建模块build -------------


#  -------------------------- 2、定义残差网络ResNet的类 ----------------

# -----------------------------3、模型调用-----------------------------
model = ResNet.build(32, 32, 3, 10, stages=[3, 4, 6], filters=[64, 128, 256, 512])
plot_model(model, to_file='resnet.png', show_shapes=True)

# -----------------------------3、模型调用-----------------------------
