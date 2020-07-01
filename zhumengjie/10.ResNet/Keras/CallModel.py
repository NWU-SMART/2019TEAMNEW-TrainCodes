# 开发者：朱梦婕
# 开发日期：2020年7月1日
# 开发框架：Keras
# 开发内容：搭建残差网络并输出模型
#----------------------------------------------------------
from ResnetModel import ResNet
from keras.utils import plot_model

# 构建残差网络
model = ResNet.build(32, 32, 3, 10, stages=[3, 4, 6], filters=[64, 128, 256, 512])  # 因为googleNet默认输入32*32的图片
# 输出模型
plot_model(model, to_file="output/resnet.png", show_shapes=True)