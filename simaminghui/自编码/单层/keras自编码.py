# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/28 002816:03
# 文件名称：keras自编码
# 开发工具：PyCharm


# minst 数据路径

import numpy
import matplotlib.pyplot as plt
from keras import Input, Model
from keras.layers import Dense

path = 'D:\DataList\mnist\mnist.npz'

# 加载数据
f = numpy.load(path)
print(dict(f).keys())  # 转成字典查看key——————dict_keys(['x_test', 'x_train', 'y_train', 'y_test'])
print(f.files),  # 以npz结尾的数据集是压缩文件，里面还有其他的文件,使用f.files进行查看，['x_test', 'x_train', 'y_train', 'y_test']

# 得到训练的图像和训练的图像
x_train, x_trest = f['x_train'], f['x_test']  # 维度为(60000, 28, 28)，(10000, 28, 28)
# 展示第一个图像，为5。因为数据已经是矩阵可之间显示
plt.imshow(x_train[0])
plt.show()
print(f['y_train'][0])

# 数据归一化
x_train = x_train.astype('float32') / 255
x_test = x_trest.astype('float32') / 255

# 将图像矩阵转为向量，
# numpy.prod(x)函数作用是将x数组的所有数字相乘，然后返回结果，比如x=[4,5,6],则返回120
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_trest.reshape(10000, 28 * 28)

# ---------------------------单层自编码器模型---------------------------------
# 定义神经网络层数
x = Input(shape=(28 * 28,))
h = Dense(64, activation='relu')(x)
r = Dense(784, activation='sigmoid')(h)
model = Model(x, r)
model.compile(optimizer='adam', loss='mse')

# ----------------------------------模型可视化-------------
from IPython.display import SVG  # SVG使得图像不会失帧
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

history = model.fit(x_train, x_train,
                    batch_size=128,
                    epochs=5,
                    validation_split=0.1)

# ------------------------------查看自编码的压缩效果------------------------
conv_encoder = Model(x, h)  # 只取编码器做模型（取出输入层x,和隐藏层h，作为网络结构）
encoded_imgs = conv_encoder.predict(x_test)


# 打印10张测试集手写体的压缩效果
n = 10
plt.figure(figsize=(20,8))
for i in range(10):
    ax = plt.subplot(1,n,i+1) # 画子图，参数（m,n,p）m表示排成m行，n表示排成多少列，p表示位置，p=1表示从上到下，从左到右的第一个位置
    plt.imshow(encoded_imgs[i].reshape(4,16).T) # 8*8 的特征，转化为 4*16的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# --------------------------------查看自编码其的解码效果------------------
decoded_imgs = model.predict(x_test)
n = 10
plt.figure(figsize=(20, 6))
for i in range(n):
    # 打印原图
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 打印解码图
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28)) # 784 转换为 28*28大小的图像
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

#-----------------------训练可视化----------------------------------
history = history.history

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()


