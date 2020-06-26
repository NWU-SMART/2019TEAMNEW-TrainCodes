#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月26日
# 开发框架：keras
# 开发内容：实现自定义contrastive loss及多输入多输出模型
#----------------------------------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入 Keras numpy, os 的包
# 2、定义输入
# 3、MIMO模型搭建
# 4、自定义contrastive loss层
# 5、模型的建立、编译及显示
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
from keras.layers import Input
from keras.models import Model
import keras
import numpy as np
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import Adam
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
#  -------------------------- 导入需要包 -------------------------------

#  -------------------------- 2、定义输入 -------------------------------
# 定义输入
img_1 = Input((48, 48, 3))
img_2 = Input((48, 48, 3))
label_1 = Input((7,))
label_2 = Input((7,))
inputs = Input((48, 48, 3))
#  -------------------------- 定义输入 -------------------------------

#  -------------------------- 3、MIMO模型搭建 -------------------------------
# 模型搭建
x = keras.layers.Conv2D(16, (3, 3), padding='SAME', activation='relu')(inputs) # 48*48*3->48*48*16
x = keras.layers.BatchNormalization()(x)

x = keras.layers.MaxPool2D((2, 2))(x)  # 48*48*3->24*24*16
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation='relu')(x) # 24*24*16->256

x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(7, activation='softmax', name='softmax')(x) # 256->7

model = Model(inputs=inputs, outputs=x)
out_1 = model(img_1)      # 输出1
out_2 = model(img_2)      # 输出2
#  -------------------------- 3、MIMO模型搭建 -------------------------------

#  -------------------------- 4、自定义contrastive loss层 -------------------------------
'''
keras 定义层的方法:
使用keras.layers.Lambda构建层func为函数  keras.layers.Lambda(func)(input),
Lambda叫做匿名函数，用它可以精简代码，匿名函数可以在程序中任何需要的地方使用，但是这个函数只能使用一次，即一次性的。
'''
# out_1与out_2之间的欧式距离（特征向量之间的距离）
# 输入：[out_1, out_2]
# 输出：eudist
eudist = keras.layers.Lambda(lambda x: KTF.cast(KTF.sum(KTF.square(x[0] - x[1]), -1, keepdims=True), np.float32))([out_1, out_2])
'''
KTF.cast: Casts a tensor to a different dtype and returns it.
'''

# label_1与label_2之间的欧式距离（判断是否为同一标签）
# 输入：[label_1, label_2]
# 输出：flag
flag = keras.layers.Lambda(lambda x: KTF.cast(
    KTF.equal(KTF.ones_like(eudist) * 7, KTF.sum(KTF.cast(KTF.equal(x[0], x[1]), np.float32), -1, keepdims=True)),
    np.float32))([label_1, label_2])
'''
KTF.equal:Element-wise equality between two tensors.
'''

# contrastive loss:
# 输入：[eudist, flag]
# 输出：loss
contrast_loss = keras.layers.Lambda(
    lambda x: KTF.mean(x[0] * x[1] + (1. - x[1]) * KTF.square(KTF.maximum(0., 1.5 - KTF.sqrt(x[0]))), 0, keepdims=True),
    name='contrast_loss')([eudist, flag])
#  -------------------------- 自定义contrastive loss层 -------------------------------

#  -------------------------- 5、模型的建立、编译及显示 -------------------------------
# 建立模型3：
# 输入：[img_1, img_2, label_1, label_2]
# 输出：[out_1, out_2, contrast_loss]
model_3 = Model(inputs=[img_1, img_2, label_1, label_2], outputs=[out_1, out_2, contrast_loss], name='s_model')
# 编译模型3
model_3.compile(optimizer=Adam(0.001),
                loss=['categorical_crossentropy', 'categorical_crossentropy', lambda y_true, y_pred: y_pred],
                loss_weights=[0.3, 0.3, 0.4], metrics=['accuracy'])
'''
lambda y_true, y_pred: y_pred：定义的匿名函数，输入(y_true, y_pred)，返回y_pred
'''

# 建立模型1、2
model_1 = Model(inputs=img_1, outputs=out_1, name='model1')
model_2 = Model(inputs=img_2, outputs=out_2, name='model2')

# 编译模型1、2
model_1.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model_2.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 显示模型1、2、3
model_1.summary()
model_2.summary()
model_3.summary()
#  --------------------------模型的建立、编译及显示 -------------------------------