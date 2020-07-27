# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/27 002716:34
# 文件名称：CallModel
# 开发工具：PyCharm
from 残差网络.ResnetModel import ResNet
from keras.utils import plot_model
model = ResNet.build(32,32,3,10,stages=[3,4,6],filters=[64,128,256,512])
# 输出模型
plot_model(model,to_file="resnet.png",show_shapes=True)