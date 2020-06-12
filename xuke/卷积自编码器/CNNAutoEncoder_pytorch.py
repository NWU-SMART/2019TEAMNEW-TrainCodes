#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.12
#software：pycharm
#项目名称：CNN自编码器（pytorch）
#--------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看解码效果
# 7、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import numpy as np
import matplotlib.pyplot as plt
from keras import Model
#  -------------------------- 1、导入需要包 -------------------------------

#-------------------2、读取手写体数据及与图像预处理-----------------------
# 载入数据
path = 'D:\\keras\\编码器\\mnist.npz'           # 数据集路径
f = np.load(path)                              # 打开文件
print(f.files)                                 # 查看文件内容 ['x_test', 'x_train', 'y_train', 'y_test']
x_train = f['x_train']                         # 定义训练数据 60000个
x_test = f['x_test']                           # 定义测试数据 10000个
f.close()                                      # 关闭文件
# 数据预处理
# 数据格式进行转换
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.

# 打印训练数据的维度
print(x_train.shape)  # (60000, 28, 28,1)
# 打印测试数据的维度
print(x_test.shape)   # (10000, 28, 28,1)
##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28, 1)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

#-----------------------2、读取手写体数据及与图像预处理--------------------
#---------------------------3.卷积自编码器模型---------------------------
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),  # 1*28*28 --> 16*28*28
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2)                           # 16*28*28 --> 16*14*14
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,8,kernel_size=3,stride=1,padding=1),  # 16*14*14 --> 8*14*14
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2)                           # 8*14*14--> 8*7*7
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(8,8,kernel_size=3,stride=1,padding=1),  # 8*7*7--> 8*7*7
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,stride=2)                          # 8*7*7--> 8*4*4
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # 8*4*4--> 8*4*4
            torch.nn.ReLU(),
            torch.nn.Upsample((8,8))
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # 8*8*8--> 8*8*8
            torch.nn.ReLU(),
            torch.nn.Upsample((16,16))                                  # 8*8*8--> 8*16*16
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),  # 8*16*16-> 16*14*14
            torch.nn.ReLU(),
            torch.nn.Upsample((28,28))                                   # 16*14*14-> 16*28*28
        )
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # 16*28*28-> 1*28*28
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        output = self.output(conv6)
        return output
model = Model()

#---------------------------3.卷积自编码器模型---------------------------
#  --------------------- 4、模型可视化 ---------------------

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
SVG(model_to_dot(model).create(prog='dot', format='svg'))

#  --------------------- 4、模型可视化 ---------------------

#  --------------------- 5、训练 ---------------------

# 设定peochs和batch_size大小
epochs = 3
batch_size = 128
history = model.fit(x_train, x_train,batch_size=batch_size,epochs=epochs, verbose=1,validation_data=(x_test, x_test))

#  --------------------- 5、训练 ---------------------

#  --------------------- 6、查看解码效果 ---------------------

# decoded_imgs 为输出层的结果
decoded_imgs = model.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 原图
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 解码效果图
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

#  --------------------- 6、查看解码效果 ---------------------


#  --------------------- 7、训练过程可视化 ---------------------

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#  --------------------- 7、训练过程可视化 ---------------------