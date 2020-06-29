# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/26 002617:06
# 文件名称：__init__.py
# 开发工具：PyCharm


# 数据集路径
import gzip
import os

import keras
import numpy
from keras import Input, Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras_preprocessing.image import ImageDataGenerator


def load_data():
    path = [
        'D:\DataList\images\\train-images-idx3-ubyte.gz',
        'D:\DataList\images\\train-labels-idx1-ubyte.gz',
        'D:\DataList\images\\t10k-images-idx3-ubyte.gz',
        'D:\DataList\images\\t10k-labels-idx1-ubyte.gz',
    ]
    # 训练标签
    with gzip.open(path[1]) as train_label:
        # np.uint8——(0-255),
        y_train = numpy.frombuffer(train_label.read(), numpy.uint8, offset=8)  # offset表示从第几位开始读入
    with gzip.open(path[0]) as train_images:
        x_train = numpy.frombuffer(train_images.read(), numpy.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(path[3]) as test_label:
        y_test = numpy.frombuffer(test_label.read(), numpy.uint8, offset=8)
    with gzip.open(path[2]) as test_images:
        x_test = numpy.frombuffer(test_images.read(), numpy.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 128
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'save_models_cnn')  # getcwd获取当前目录
model_name = 'keras_fashion_trained_model.h5'

# one-hot编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)  # 若num——classes不设置，会自动识别
# (10000, 10) [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
print(y_test.shape, y_test[0])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255  # 归一化
x_test = x_test / 255

#--------------------------API---------------------------------------
inputs = Input(shape=x_train.shape[1:])  # 也就是 28*28*1
x = Conv2D(32, (3, 3), padding='same')(inputs)  # 32表示卷积核数量（输出深度），3*3表示卷积核大小，same表示输出的长和宽和输入一样
x = Activation('relu')(x)  # 进行relu激活
x = Conv2D(32, (3, 3))(x)  # 继续卷积，padding填充默认为 valid，表示不填充
x = Activation('relu')(x)  # relu激活
x = MaxPooling2D(pool_size=(2, 2))(x)  # 矩阵缩小一半，即矩阵中4个元素选取最大一个元素
x = Dropout(0.25)(x)  # 防止过拟合

x = Conv2D(64, (3, 3), padding='same')(x)  # 卷积
x = Activation('relu')(x)  # 激活
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)  # 为进入全连接层做准备
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# 优化函数
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['acc']

)

# ---------------------------------------------训练---------------------------------
# 如果没有图像增强
if not data_augmentation:
    print('Not Using data augmentation.')
    # 直接进行训练
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True  # 每轮迭代之前混洗数据
                        )

else:  # 概念查看（该目录下部分概念原理.md）
    print('Using real-time data augmentation')
    datagen = ImageDataGenerator(
        # feature 是对整个数据集来操作的
        featurewise_center=False,  # 使输入数据去中心化(均值为0),使数据集去中心化
        featurewise_std_normalization=False,  # 输入数据处于数据集的标准差，完成标准化

        # sample 是对单个样本的操作
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差

        # 白化原理概念（部分概念原理.md）
        zca_whitening=False,  # 对输入数据施加ZCA白化

        # 不是固定旋转，而是[0,指定角度]范围内进行随机角度旋转
        rotation_range=0,  # 数据提升时图片随机转动的角度。随机选择图片的角度，是一个0~180的度数，取值为0~180。

        # 水平位置平移和上下位置平移 水平平移距离为宽度*参数，上下平移距离为高度*参数，同样平移距离不固定，为[0,x]区间的任意数
        width_shift_range=0.1,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
        height_shift_range=0.1,  # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度

        shear_range=0.,  # 浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度
        zoom_range=0.,  # 当给出一个数时，图片同时在长宽两个方向进行同等程度的放缩操作，参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作

        # 可以理解成改变图片的颜色，通过对颜色通道的数值偏移，改变图片的整体的颜色，这意味着是“整张图”呈现某一种颜色，
        # 像是加了一块有色玻璃在图片前面一样，因此它并不能单独改变图片某一元素的颜色，如黑色小狗不能变成白色小狗。
        # 当数值越大时，颜色变深的效果越强
        channel_shift_range=0.,

        # 填充模式，当对图片平移，放缩等操作时，图片中会出现缺失的地方，缺失的地方就用fill——mode补全
        # “constant”、“nearest”（默认）、“reflect”和“wrap”四种填充方式，c
        fill_mode='nearest',
        cval=0.,  # 当fill_mode为constant时，使用某个固定数值的颜色来进行填充

        horizontal_flip=True,  # 水平翻转
        vertical_flip=False,  # 随机垂直翻转

        # rescale的作用是对图片的每个像素值均乘上这个放缩因子，这个操作在所有其它变换操作之前执行，
        # 如：rescale= 1/255
        rescale=None,
        preprocessing_function=None,

        validation_split=0.0
    )
    datagen.fit(x_train)

    # ----------------------------------------------训练------------------------------
    # 数据增强后采用fit_generator的方法训练

    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),  # 按batch_size大小从x,y生成增强数据
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量

                                  )
# ------------------------------------------保存模型-------------------------------------
model.summary()

if not os.path.isdir(save_dir):  # 如果目录不存在
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print("模型保存在 %s" % model_path)

#-----------------------------------------显示结果------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model loss")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("model acc")
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()