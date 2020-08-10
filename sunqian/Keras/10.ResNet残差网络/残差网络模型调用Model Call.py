# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/8/9
# 文件名称：残差网络模型ResnetModel.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#
from 残差网络模型ResnetModel import ResNet
from keras.utils import plot_model

# 构建残差网络
model = ResNet.build(32, 32, 3, 10, stages=[3, 4, 6], filters=[64, 128, 256, 512])  # 因为googleNet默认输入32*32的图片
# 输出模型
plot_model(model, to_file="resnet.png", show_shapes=True)