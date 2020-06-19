# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.18
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/
from keras.models import Sequential
from keras.layers import merge, Conv2D, MaxPool2D, Activation, Dense, concatenate, Flatten
from keras.layers import Input
from keras.models import Model
from keras.utils.vis_utils import plot_model

# ---------------------函数功能区-------------------------
def model(inp):


    models = Conv2D(filters=24, kernel_size=(3, 3), strides=1, padding='same')(inp)
    models = Activation('relu')(models)
    models = MaxPool2D(pool_size=(3, 3))(models)

    models = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')(models)
    models = Activation('relu')(models)
    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)

    models = Conv2D(filters=96, kernel_size=(3, 3), strides=1, padding='valid')(models)
    models = Activation('relu')(models)


    models = Flatten()(models)
    models = Dense(512)(models)
    models = Activation('relu')(models)
    return models



input_1 = Input(shape = (28, 28, 1))
input_2 = Input(shape = (28, 28, 1))
result_1 = model(input_1)
result_2 = model(input_2)
# ----- 两个网络层向量拼接
merge_layers = concatenate([result_1, result_2])        # 进行融合，使用的是默认的sum，即简单的相加

# ----- 全连接
fc1 = Dense(1024, activation='relu')(merge_layers)
fc2 = Dense(256, activation='relu')(fc1)
fc3 = Dense(2, activation='softmax')(fc2)

# 构建最终网络
class_models = Model(inputs=[input_1, input_2], outputs=[fc3])

plot_model(class_models, to_file='model_孪生.png',show_shapes=True)  # 网络结构输出成png图片

