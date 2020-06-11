from gaoqiang.ResNet.keras_resnet import *   # 注意精确到根工程
from keras.utils import plot_model

model = ResNet.build(32,32,3,10,stages=[3,4,6],filters=[64,128,256,512]) # 输入32*32的图片
# 保存网络结构图
plot_model(model, to_file="keras_resnet.png", show_shapes=True)

'''
1.stage1的范围：BN2~add_3

conv2d_5是stage1当中的第一个residual_module函数中shortcut降维，然后到add_3位置，是stage1的reduce=False非降维部分

2.stage2的范围：BN11~add_7

conv2d_15是stage2当中的第一个residual_module的shortcut降维

3.stage3的范围：BN23~add_13

conv2d_28是stage3当中的第一个residual_module的shortcut降维

'''