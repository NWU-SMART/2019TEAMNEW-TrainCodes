from cancha.resnet import *   # 注意精确到根工程
from keras.utils import plot_model

model = ResNet.build(32,32,3,10,stages=[3,4,6],filters=[64,128,256,512]) # 输入32*32的图片
# 保存网络结构图
plot_model(model, to_file="keras_resnet.png", show_shapes=True)

