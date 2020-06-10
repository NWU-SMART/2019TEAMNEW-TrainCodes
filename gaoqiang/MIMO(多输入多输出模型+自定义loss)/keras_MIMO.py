# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.09
# 开发框架：keras
# 代码功能：MIMO:多输入多输出模型，自定义loss
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局------------------------------------------------------------#
# 1、建立模型
# 2、保存模型与模型可视化
# 3、自定义loss
#----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------建立模型------------------------------------------------------------#
# 定义输入
from keras.layers import Input
input_tensor1 = Input((32,32,3))
input_tensor2 = Input((4,))
input_target = Input((2,))

# input_tensor1  支路
import keras
x = keras.layers.BatchNormalization(axis=-1)(input_tensor1) # 正则

x = keras.layers.Conv2D(16,(3,3),padding="same")(x)         # 卷积(32,32,3------32,32,16)
x = keras.layers.Activation("relu")(x)                      # 激活
x = keras.layers.MaxPooling2D(2)(x)                         # 池化(32,32,16------16,16,16)

x = keras.layers.Conv2D(32,(3,3),padding="same")(x)         # 卷积(16,16,16------16,16,32)
x = keras.layers.Activation("relu")(x)                      # 激活
x = keras.layers.MaxPooling2D(2)(x)                         # 池化(16,16,32-----8,8,32)

x =  keras.layers.Flatten()(x)
x = keras.layers.Dense(32)(x)
x = keras.layers.Dense(2)(x)

out2 = x                                                    # 命名 input_tensor1 对应 out2

# input_tensor2  支路
y = keras.layers.Dense(32)(input_tensor2)
y = keras.layers.Dense(2)(y)

out1 = y


# 定义模型
from keras import Model
model_temp = Model([input_tensor1,input_tensor2,input_target],[out1,out2])

#----------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------模型可视化及保存模型图片--------------------------------------------------#
model_temp.summary()
# 可视化模型(保存模型图片)
from keras.utils.vis_utils import plot_model
plot_model(model_temp,to_file="model_temp.png",show_shapes=True)
#----------------------------------------------------------------------------------------------------------------------#

#--------------------------------------------------自定义loss----------------------------------------------------------#
'''
keras 定义层的方法:
使用keras.layers.Lambda构建层func为函数  keras.layers.Lambda(func)(input),
Lambda叫做匿名函数，用它可以精简代码，匿名函数可以在程序中任何需要的地方使用，但是这个函数只能使用一次，即一次性的。
'''
import keras.backend as K                        # 后端backend
def custom_loss1(y_true,y_pred):                 # 自定义平均绝对误差
    return K.mean(K.abs(y_true-y_pred))

loss1 =  keras.layers.Lambda(lambda x:custom_loss1(*x),name='loss1')([out2,out1])
loss2 =  keras.layers.Lambda(lambda x:custom_loss1(*x),name='loss2')([input_target,out2])
model = Model([input_tensor1,input_tensor2,input_target],[out1,out2,loss1,loss2])
plot_model(model,to_file="model.png",show_shapes=True)
#----------------------------------------------------------------------------------------------------------------------#
# 取出loss层model.get_layer() 并取出loss层的输出 model.get_layer(name).output
loss_layer1 = model.get_layer("loss1").output
loss_layer2 = model.get_layer("loss2").output
model.add_loss(loss_layer1)
model.add_loss(loss_layer2)
model.compile(optimizer='sgd',loss=[None,None,None,None])
history = model.fit()


