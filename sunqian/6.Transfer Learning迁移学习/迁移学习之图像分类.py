# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/14
# 文件名称：迁移学习之图像分类.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入数据、数据预处理
# 3、构建模型
# 4、训练模型并保存
# 5、训练可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import gzip
import numpy as np
import keras
from keras import applications

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import os
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入数据、数据预处理 -------------------------------------------
path = 'E:\\keras_datasets\\'
def load_data():
    paths = [
        path+'train-labels-idx1-ubyte.gz',  path+'train-images-idx3-ubyte.gz',
        path+'t10k-labels-idx1-ubyte.gz',  path+'t10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0],'rb') as lbpath: #'rb'指读取二进制文件，非人工书写的数据如.jpg等
        y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)

    with gzip.open(paths[1],'rb') as imgpath:
        # frombuffer将data以流的形式读入转化成ndarray对象
        # 第一参数为stream，第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(),np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8,offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()

batch_size = 32
num_classes = 10
epochs = 10
data_augmentation = True  # 图像增强
num_prediction = 20
# os.getcwd()返回当前进程的工作目录
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')  #os.path.join()函数连接两个或更多的路径名组件
model_name = 'keras_fashion_transfer_learning_model.h5'

# 将类别转换成独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 由于mnist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,所以扩充mnist的最后一维
import cv2
x_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化
x_train /=255
x_test /=255
print(x_train.shape)  # 结果为(60000, 48, 48, 3)
#  -------------------------- 2、导入数据、数据预处理--------------------------------------------

#  -------------------------- 3、迁移学习建模 -------------------------------------------
# 使用vgg16模型，include_top=False表示不包含最后的3个全连接层。input_shape=x_train.shape[1:]第一层需要指出图像大小
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])

# 建立CNN模型
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))

# 7*7*512 --> 256
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))

# 256 -->num_classes
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 输入为VGG16的数据，经过VGG16的特征层，2层全连接到num_classes输出
model = Model(inputs=base_model.input, outputs=model(base_model.output))

# 保持VGG16的前15层权值不变，即在训练过程中冷冻前15层
for layer in model.layers[:15]:
    layer.trainable = False

# 初始化RMSprop优化器
opt = RMSprop(lr=0.0001, decay=1e-6)  # lr指学习率learning rate decay指学习率每次更新的下降率
# 使用RMSprop优化器
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
#  -------------------------- 3、迁移学习建模------------------------------------------

#  -------------------------- 4、训练模型------------------------------------------
# 如果没有图像增强就直接训练
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train,y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test,y_test),
                        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # 数据预处理与实时数据增强
    datagen = ImageDataGenerator(
        # feature针对整个数据集
        featurewise_center=False,  # 使输入数据，即数据集去中心化（均值为0）
        featurewise_std_normalization=False,  # 输入数据处于数据集的标准差，完成标准化

        # sample针对单个样本
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        samplewise_std_normalization=False,  # 将输入的每个样本除以自身的标准差

        zca_whitening=False,  # 对输入数据施加ZCA白化
        zca_epsilon=1e-06,

        rotation_range=0,  # 数据增强时图片随机转动的角度，角度0—180
        # 水平位置平移和上下位置平移
        width_shift_range=0.1,  # 数据增强时图片随机水平偏移的幅度
        height_shift_range=0.1,  # 数据增强时图片随机竖直偏移的幅度

        shear_range=0,  # 剪切强度（逆时针方向的剪切变换角度）

        zoom_range=0,  # 当给出一个数时，图片同时在长宽两个方向进行同等程度的放缩操作，参数大于0小于1时，执行的是放大操作，当参数大于1时，执行的是缩小操作

        channel_shift_range=0.,  # 改变图片的颜色，通过对颜色通道的数值偏移，改变图片的整体的颜色

        # “constant”、“nearest”（默认）、“reflect”和“wrap”四种填充方式
        fill_mode='nearest',  # 填充模式，当对图片平移，放缩等操作时，图片中会出现缺失的地方，缺失的地方就用fill——mode补全
        cval=0.,   # 当fill_mode为constant时，使用某个固定数值的颜色来进行填充

        horizontal_flip=True,  # 水平翻转
        vertical_flip=False,  # 随机垂直翻转

        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # 拟合模型
    # 按batch_size大小从x,y生成增强数据
    # flow_from_directory()从路径生成增强数据，和flow方法相比最大的优点在于不用一次性将所有数据读入内存中，可以减小内存压力
    history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        workers=10) # 在使用基于进程的线程时，最多需要启动的进程数量

# 保存模型
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir,model_name)
model.save(model_path)
print('Saved trained model at %s '%model_path)

#  -------------------------- 4、训练模型并保存------------------------------------------

#  -------------------------- 5、训练可视化-------------------------------------------
# 绘制训练和验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

# 绘制训练和验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
# 训练精度可达到85%，验证精度可达到87%
#  -------------------------- 5、训练可视化 ------------------------------------------
