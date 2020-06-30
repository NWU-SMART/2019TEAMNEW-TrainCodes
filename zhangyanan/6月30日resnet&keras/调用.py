# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月30日
# 修改日期：
# 修改人：
# 修改内容：
from cancha.resnet import *   # 注意精确到根工程
from keras.utils import plot_model

model = ResNet.build(32,32,3,10,stages=[3,4,6],filters=[64,128,256,512]) # 输入32*32的图片
# 保存网络结构图
plot_model(model, to_file="keras_resnet.png", show_shapes=True)

