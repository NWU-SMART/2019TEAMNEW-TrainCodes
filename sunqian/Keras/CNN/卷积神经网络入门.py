# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/28
# 文件名称：卷积神经网络入门.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入mnist数据
# 3、构建网络
# 4、模型训练
# 5、模型预测
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入mnist数据 -------------------------------
# 数据存在本地路径E:\\keras_datasets\\mnist.npz
path='E:\\keras_datasets\\mnist.npz'
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path)

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#  -------------------------- 2、导入mnist数据-------------------------------

#  -------------------------- 3、构建网络-------------------------------
model=models.Sequential()
# 构建一个简单的卷积神经网络，它是Conv2D层和MaxPooling2D层的堆叠
model.add(layers.Conv2D(32, (3, 3), activation='relu',  input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# 在卷积神经网络上添加分类器
# 将最后的输出张量（大小为(3,3,64)）输入到一个密集连接分类器网络中，即Dense层的堆叠。
# 这些分类器可以处理1D向量，而当前的输出是3D张量，因此先将3D输出展平为1D,然后在上面添加几个Dense层
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()  # 查看模型 在进入两个Dense层之前，形状(3,3,64)的输出被展平为形状(576,)的向量

#  -------------------------- 3、构建网络-------------------------------

#  -------------------------- 4、模型训练-------------------------------
# 在mnist图像上训练卷积神经网络
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
#  -------------------------- 4、模型训练-------------------------------

#  -------------------------- 5、模型预测-------------------------------
test_loss,test_acc= model.evaluate(test_images,test_labels)
test_acc
# 结果为0.9919999837875366 即这个简单的卷积神经网络的测试精度达到了99.2% 效果很好

#  -------------------------- 5、模型预测 -------------------------------

