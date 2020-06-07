# ----------------开发者信息-------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月3日
# 开发框架：keras
#---------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 keras, numpy, functools, os 和 gzip的包
# 2、参数定义
# 3、读取数据与数据预处理
# 4、搭建CNN模型
# 5、训练
# 6、保存模型
# 7、显示运行结果
#--------------------------------------------------------------#

#  -------------------------- 1、导入需要包 -------------------------------
from tensorflow.python.keras.utils import get_file
import gzip
import keras
import numpy as np
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,4'
# 使用四块显卡进行加速
#  --------------------------导入需要包 -------------------------------

#  -------------------------- 2、参数定义 -------------------------------
batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'
#  -------------------------- 参数定义 -------------------------------

#  -------------------------- 3、读取数据与数据预处理 -------------------------------
# 函数：数据加载
def load_data():
    # 写入文件路径
    paths = [
        'E:\\study\\mnist\\train-labels-idx1-ubyte.gz', 'E:\\study\\mnist\\train-images-idx3-ubyte.gz',
        'E:\\study\\mnist\\t10k-labels-idx1-ubyte.gz', 'E:\\study\\mnist\\t10k-images-idx3-ubyte.gz'
    ]

    # 将文件解压并划分为数据集
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data() # 加载数据集

# 将类型信息进行one-hot编码(10类)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 将图片信息转换数据类型
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化
x_train /= 255
x_test /= 255
#  -------------------------- 读取数据与数据预处理 -------------------------------

#  -------------------------- 4、搭建CNN模型 -------------------------------
input = Input(shape=x_train.shape[1:])
x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=input,outputs=x)
# 优化器初始化
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#  --------------------------  搭建CNN模型  -------------------------------

#  -------------------------- 5、训练 -------------------------------
# 通过判断是否使用图片增强技术去训练图片
if not data_augmentation:
    print('没有使用图像增强技术')
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                        shuffle=True)
else:
    print('使用了图像增强技术')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 将输入数据的均值设为0
        samplewise_center=False,  # 将每个样本的均值设为0
        featurewise_std_normalization=False,  # 逐个特征将输入数据除以标准差
        samplewise_std_normalization=False,  # 将每个输入除以标准差
        zca_whitening=False,  # 是否使用zca白化（降低输入的冗余性）
        zca_epsilon=1e-06,  # 利用阈值构建低通滤波器对输入数据进行滤波
        rotation_range=0,  # 随机旋转的度数
        width_shift_range=0.1,  # 水平随机移动宽度的0.1
        height_shift_range=0.1, # 垂直随机移动高度的0.1
        shear_range=0.,  # 不随机剪切
        zoom_range=0.,  # 不随机缩放
        channel_shift_range=0.,  # 通道不随机转换
        fill_mode='nearest',  # 边界以外的点的填充模式：aaaaaaaa|abcd|dddddddd靠近哪个就用哪个填充，如靠近a那么就用a填充，靠近d就用d填充
        cval=0.,  # 边界之外点的值
        horizontal_flip=True,  # 随机水平翻转
        vertical_flip=False,  # 不随机垂直翻转
        rescale=None,        # 不进行缩放（若不为0和None将数据乘以所提供的值）
        preprocessing_function=None,  # 应用于输入的函数
        data_format=None,         # 图像数据格式
        validation_split=0.0)    # 用于验证图像的比例

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,  # 每个批次训练的样本
                        validation_data=(x_test, y_test),  # 验证集选择
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )


#  -------------------------- 训练 -------------------------------

#  -------------------------- 6、保存模型 -------------------------------

model.summary()
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#  --------------------------保存模型 -------------------------------

#  -------------------------- 7、显示运行结果 -------------------------------

import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_accuracy.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()

#  -------------------------- 保存模型，显示运行结果 -------------------------------