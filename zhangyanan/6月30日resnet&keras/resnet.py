
# ----------------------   代码布局： ----------------------
# 1、导入 Keras的包
# 2、定义残差网络ResNet的类
#      定义ResNet的残差模块residual_module
#      定义ResNet的网络构建模块build
# ----------------------   代码布局： ----------------------

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
    # @staticmethod装饰器：一般在写一个方法的时候, 默认会接受一个self的形参, 但是在调用这个方法的使用可能并没有传递任何一个参数,
    # 这个self就是你使用对象调用方法的那个对象本身,要是将这个方法改为一个静态的方法, 就不会有self
    # 当某个方法不需要用到对象中的任何资源,将这个方法改为一个静态方法, 加一个@staticmethod
    # 加上之后, 这个方法就和普通的函数没有什么区别了, 只不过写在了一个类中, 可以使用这个类的对象调用,也可以使用类直接调用,
    @staticmethod
    def residual_module(x, K, stride, chanDim, reduce=False, reg=1e-4, bnEps=2e-5, bnMom=0.9)  :  # 结构参考Figure 12.3右图,引入了shortcut概念，是主网络的侧网络
        """
        The residual module of the ResNet architecture.
        Parameters:
            x: residual module的输入
            K: 最终卷积层的输出
            stride: 步长
            chanDim: 定义将执行批归一化的轴。
            reduce:  (True) or not (False).是否降维，
            reg: 控制残差模块中所有CONV层的正则化强度。
            bnEps: 防止BN层出现除以0的异常
            bnMom: 控制移动平均线的动量。  ？？？什么意思
        Return:
            x: 返回残差模块的输出.
        """

        # 获取的结果为F(x)+x,这里输入的信号为x(直接跳转)，F(x)为经过多个卷积处理函数
        # F(x)经过的卷积操作为1x1（第一层），3x3（第二层）和1x1（第三层）
        shortcut = x

        # ------ 第一层卷积(归一、激活和1x1卷积)
           # 归一
        bn1   = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
           # 激活
        act1  = Activation("relu")(bn1)
           # 1x1卷积
        conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg)) \
            (act1  )  # filter=K*0.25,kernel_size=(1,1),stride=(1,1)

        # ------ 第二层卷积(归一、激活和1x1卷积)
           # 归一
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
           # 激活
        act2 = Activation("relu")(bn2)
           # 3x3卷积
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, kernel_regularizer=l2(reg))(act2)

        # ------ 第三层卷积(归一、激活和1x1卷积)
          # 归一
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
          # 激活
        act3 = Activation("relu")(bn3)
          # 1x1卷积
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        if reduce  :  # 是否降维，如果降维的话，需要将stride设置为大于1,更改shortcut值
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        # 计算F(x)+x
        x = add([conv3, shortcut]  )  # 这个与googlenet的concatenate函数不同，add函数做简单加法，concatenate函数做横向拼接.该函数仅仅将shortcut部分和非shortcut部分相加在一起

        # 函数返回F(x)+x
        return  x  # f(x)输出结果=conv3+shortcut

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
        # ----- 1、按照层次数构建残差网络（多少个blok，每个block中有多少个残差块）
        # ----- 2、构建全连接网络，进行分类

        # ----- 1、按照层次数构建残差网络（多少个blok，每个block中有多少个残差块）
        # 构建多层残差网络（len(stages)）
        # stages=[3, 4, 6] --> 3个block，每个block中分别有3,4,6个残差块
        for i in range(0, len(stages)):  # 每阶段的遍历

            # --- 初始步长，Initialize the stride, then apply a residual module used to reduce the spatial size of the input volume.

            # If this is the first entry in the stage, we’ll set the stride to (1, 1), indicating that no downsampling
            # should be performed. However, for every subsequent stage we’ll apply a residual module with a stride of (2, 2),
            # which will allow us to decrease the volume size.
            # 构建第一个block，则设置步长为 (1,1)，否则为(2,2)
            stride = (1, 1) if i == 0 else (2, 2)

            # Once we have stacked stages[i] residual modules on top of each other, our for loop brings us back up to here
            # where we decrease the spatial dimensions of the volume and repeat the process.
            x = ResNet.residual_module(x, filters[i + 1], stride=stride, chanDim=chanDim, reduce=True, bnEps=bnEps,
                                       bnMom=bnMom)  # 进行降维

            # Loop over the number of layers in the stage.
            for j in range(0, stages[i] - 1):  # 每层的遍历
                # Apply a residual module.
                x = ResNet.residual_module(x, filters[i + 1], stride=(1, 1), chanDim=chanDim, bnEps=bnEps,
                                           bnMom=bnMom)  # 不进行降维
        # ----- 1、按照层次数构建残差网络（多少个blok，每个block中有多少个残差块）

        # ----- 2、构建全连接网络，进行分类
        # 在构建深度残差网络后，构建分类器
          # 归一化
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
          # 激活
        x = Activation("relu")(x)
          # 平均池化
        x = AveragePooling2D((8, 8))(x)

        # Softmax classifier.
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation("softmax")(x)
        # ----- 2、构建全连接网络，进行分类

        # 构建模型
        model = Model(input, x, name="ResNet")

        model.summary()  # 输出网络结构信息
        # plot_model(model, to_file='./output/resnet_visualization.png', show_shapes=True, show_layer_names=True)
        # Return the build network architecture.
        return model
    # ------------------ 定义ResNet的网络构建模块build -------------

#  -------------------------- 2、定义残差网络ResNet的类 -------------------------------
