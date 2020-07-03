# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年7月2日
# 内容:Resnet
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 Keras的包
# 2、定义残差网络ResNet的类
#      定义ResNet的残差模块residual_module
#      定义ResNet的网络构建模块build
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers import Input, add
from keras.models import Model
from keras.regularizers import l2  # 这里加入l2正则化目的是为了防止过拟合
from keras.utils.vis_utils import plot_model
import keras.backend as K#如果你想用Keras写出兼容theano和tensorflow两种backend的代码，那么
# 必须使用抽象keras backend API来写代码。如何才能使用抽象Keras backend API呢，就是通过导入模块。
#导入后K模块提供的所有方法都是abstract keras backend API
#  ---------------------- 1、导入需要包 -----------------------------------
# resnet做加和操作，因此用add函数，
# googlenet以及densenet做filter的拼接，因此用concatenate
'''
concatenate操作是网络结构设计中很重要的一种操作，经常用于将特征联合，
多个卷积特征提取框架提取的特征融合或者是将输出层的信息进行融合，而add层更像是信息之间的叠加。
Resnet是做值的叠加，通道数是不变的，DenseNet是做通道的合并。
add是描述图像的特征下的信息量增多了，但是描述图像的维度本身并没有增加，只是每一维下的信息量在增加，这显然是对最终的图像的分类是有益的。
而concatenate是通道数的合并，也就是说描述图像本身的特征增加了，而每一特征下的信息是没有增加。
在代码层面就是ResNet使用的都是add操作，而DenseNet使用的是concatenate。

'''
#  -------------------------- 2、定义残差网络ResNet的类 -------------------------------
class ResNet():
      # ------------------ 定义ResNet的残差模块residual_module -------------
      @staticmethod #静态方法，类和实例都可以调用，静态方法不需要传入self
      def residual_module(x,K,stride,chanDim,reduce=False,reg=1e-4,bnEps=2e-5,bnMom=0.9):
          '''
          :param x: 残差模型的输入
          :param K:最终卷积层的输出通道
          :param stride:步长
          :param chanDim:定义批归一化处理的axis
          :param reduce:是否降维处理
          :param reg:正则化强度
          :param bnEps:防止bn层出现除以0的异常
          :param bnMom:e控制移动平均线的动量。
          :return:
          x :返回残差模块的输出
          获取的结果为F(x)+x,这里输入的信号为x(直接跳转)，F(x)为经过多个卷积处理函数
           F(x)经过的卷积操作为1x1（第一层），3x3（第二层）和1x1（第三层）
          '''
          short_cut = x
          # ------ 第一层卷积(归一、激活和1x1卷积)
          bn1=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(x)
          act1=Activation('relu')(bn1)
          conv1=Conv2D(int(K*0.5),(1,1),use_bias=False,kernel_regularizer=l2(reg))(act1)# filter=K*0.25,kernel_size=(1,1),stride=(1,1)
          ## 因为偏差在紧跟着卷积的BN层中，所以没有必要引入#第二个* bias项，因此改变了典型的CONV块顺序
          # ------ 第二层卷积(归一、激活和1x1卷积)
          bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
          act2 = Activation('relu')(bn2)
          conv2 = Conv2D(int(K * 0.5), (3, 3),strides=stride, padding="same",use_bias=False, kernel_regularizer=l2(reg))(act2)
          # ------ 第三层卷积(归一、激活和1x1卷积)
          bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
          act3 = Activation('relu')(bn3)
          conv3 = Conv2D(K , (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)
          if reduce:  # 是否降维，如果想降维的话，需要将stride设置为大于1,更改shortcut值
              shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)
          x=add([conv3,short_cut])
          return x

      # ------------------ 定义ResNet的残差模块residual_module -------------

      # ------------------ 定义ResNet的网络构建模块build -------------------
      @staticmethod #静态方法
      def bulid(width,helight,depth,classes,stages,filters,reg=1e-4,bnEps=2e-5,bnMom=0.9,dataset='cifar'):
          inputShape=(helight,width,depth)#定义输入图片的高，宽，深
          chanDim=-1
          if K.image_data_format=="channel_first":
              inputShape=(depth,helight,width)
              chanDim=1
              input=Input(shape=inputShape)
              x=BatchNormalization(axis=chanDim,epsilon=bnEps,momentum=bnMom)(input)
              # --- 归一化在这里第一层使用BN层而不是使用conv，这样可以替代取平均值的操作
          if dataset=='cifar':#是否再cifar数据集上训练
              x = Conv2D(filters[0], (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
          # ----- 步骤
          # ----- 1、按照层次数构建残差网络（多少个blok，每个block中有多少个残差块）
          # ----- 2、构建全连接网络，进行分类


      # ----- 1、按照层次数构建残差网络（多少个blok，每个block中有多少个残差块）
      # 构建多层残差网络（len(stages)）
      # stages=[3, 4, 6] --> 3个block，每个block中分别有3,4,6个残差块
          for i in range(0,len(stages)):#每个阶段的遍历
              stride=(1,1) if i==0 else (2,2)  # 第一层步长设置为1，其余层为
              x=ResNet.residual_module(x,filters[i+1],stride=stride,chanDim=chanDim,reduce=True,bnEps=bnEps,bnMom=bnMom)#进行降维
              for j in range(0,stages[i]-1):#每层遍历
                  x = ResNet.residual_module(x, filters[i + 1], stride=stride, chanDim=chanDim, bnEps=bnEps, bnMom=bnMom)  # 不进行降维
      # ----- 2、构建全连接网络，进行分类，在构建深度残差网络后，构建分类器
          x = BatchNormalization(axis=chanDim,epsilon=bnEps,bnMom=bnMom)#归一化
          x = Activation('relu')(x)#激活
          x = AveragePooling2D((8,8))(x)#平均池化
          #分类器
          x=Flatten()(x)
          x=Dense(classes,kernel_regularizer=l2(reg))(x)
          x=Activation('softmax')(x)
      #构建模型
          model=Model(input,x,name="Resnet")
          model.summary()
          plot_model(model,to_file='D:/northwest/小组视频model.png')
          return model
#  -------------------------- 2、定义残差网络ResNet的类 -------------------------------





