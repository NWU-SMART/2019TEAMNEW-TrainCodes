# ----------------------------------------------开发者信息---------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.04
# 开发框架：keras
# 温馨提示：服务器上跑
#------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局--------------------------------------------------------#
# 1、加载图像数据
# 2、图像数据预处理
# 3、训练模型
# 4、保存模型与模型可视化
# 5、训练过程可视化
#-------------------------------------------------------------------------------------------------------------------#
# 任务介绍：
'''
图像分类：基于fashion MNIST数据的图像分类去做实验。在2017年8月份，德国研究机构ZalandoResearch在GitHub上推出了一个全新的
数据集，其中训练集包含60000个样例，测试集包含10000个样例，分为10类，每一类的样本训练样本数量和测试样本数量相同。样本都
来自日常穿着的衣裤鞋包，每个都是28×28的灰度图像，其中总共有10类标签，每张图像都有各自的标签。
'''
#  -------------------------------------------- 导入需要包----------------------------------------------------------#
import gzip   # 使用python gzip库进行文件压缩与解压缩
import numpy as np
import keras

#--------------------------------------------------------------------------------------------------------------------#

# ----------------------------------------------加载数据图像---------------------------------------------------------#
def load_data():
    # 训练标签 训练图像 测试标签 测试图像
    # 本地
    # paths = [
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\train-labels-idx1-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\train-images-idx3-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-labels-idx1-ubyte.gz',
    #     'F:\\Keras代码学习\\keras\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    # ]
    # 服务器
    paths = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
    ]
    # 读取训练标签(解压)
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取训练图像(解压)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    # 读取测试标签(解压)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 读取测试图像(解压)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

# 调用函数 获取训练数据和测试数据
(x_train, y_train), (x_test, y_test) = load_data()

batch_size = 32            # 设置批大小为32
epochs = 1     # 5            # 为了节省等待时间，先设置成1个epoch
data_augmentation = True   # 使用图像增强
num_predictions = 20

import os
save_dir = os.path.join(os.getcwd(), 'saved_models_keras_transferlearning') # 保存的模型路径
model_name = 'keras_fashion_transferlearning_trained_model.h5'            # 模型名字
#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------图像数据预处理--------------------------------------------------------#
#  将整型的类别标签转为onehot编码
'''
One-Hot编码是分类变量作为二进制向量的表示。这首先要求将分类值映射到整数值。然后，每个整数值被表示为二进制向量，除了整数
的索引之外，它都是零值，它被标记为1。one-hot编码要求每个类别之间相互独立，如果之间存在某种连续型的关系，或许使用
distributed respresentation（分布式）更加合适。
'''
num_classes = 10           # 分为10类
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 由于mnist的输入数据维度是(num，28,28)，vgg16需要三维图像，因此扩充mnist的最后一维
import cv2
'''
cv2.resize：将原图放大到48*48
cv2.cvtColor(p1,p2) ：是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_GRAY2RGB 将灰度图片转化为格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
'''
x_train = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_test]

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 数据归一化
x_test /= 255   # 数据归一化

#----------------------------------------------------------------------------------------------------------------------#

#  ------------------------------------- 构建transfer leearning模型----------------------------------------------------#
'''
# 关于vgg16:
VGG16共包含：
13个卷积层（Convolutional Layer），分别用conv3-XXX表示
3个全连接层（Fully connected Layer）,分别用FC-XXXX表示
5个池化层（Pool layer）,分别用maxpool表示
其中，卷积层和全连接层具有权重系数，因此也被称为权重层，总数目为13+3=16，这即是VGG16中16的来源。(池化层不涉及权重，因此
不属于权重层，不被计数)。
优缺点：
VGG16的参数数目非常大，可以预期它具有很高的拟合能力；但同时缺点也很明显：即训练时间过长，调参难度大。需要的存储容量大，
不利于部署。例如存储VGG16权重值文件的大小为500多MB，不利于安装到嵌入式系统中。
'''
from keras import applications
# Keras 的应用模块（keras.applications）提供了带有预训练权值的深度学习模型，这些模型可以用来进行预测、特征提取和微调
# （fine-tuning）。
# 当你初始化一个预训练模型时，会自动下载权重到 ~/.keras/models/ 目录下。
'''
参数设置：
1.include_top: 是否包括顶层的全连接层。（我们这里不使用vgg16的最后三个全连接层）
2.weights: None 代表随机初始化， 'imagenet' 代表加载在 ImageNet 上预训练的权值。
3.input_shape: 可选，输入尺寸元组，仅当 include_top=False 时有效，否则输入形状必须是 (244, 244, 3)（对于 channels_last 
数据格式），或者 (3, 244, 244)（对于 channels_first 数据格式）。它必须拥有 3 个输入通道，且宽高必须不小于 32。
例如 (200, 200, 3) 是一个合法的输入尺寸。
'''
base_model = applications.VGG16(include_top = False,weights = 'imagenet',input_shape = x_train.shape[1:]) # 第一层需要指出图像的大小
print(x_train.shape[1:])  # 1875

# 连接vgg16和我们自己的模型

from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Flatten,Dropout
model = Sequential()
print(base_model.output)   # 1875.0
model.add(Flatten(input_shape = base_model.output_shape[1:]))# 拉平
# 经过vgg16后，图像参数变为 7*7*512，将其加入到我们自己的全连接结构中
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))# 输出层是10类
model.add(Activation('softmax'))
# 合并vgg16和自己的两层全连接
model = Model(input=base_model.input,outputs= model(base_model.output))

# 保持搭建的网络的前15层（共15层）权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False

#---------------------------------------------------------------------------------------------------------------------#
#  使用RMSprop优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy', # 交叉熵函数
              optimizer=opt,                   # 使用之前定义的优化器
              metrics=['accuracy'])
#  -------------------------------------------------数据增强----------------------------------------------------------#
# 判断是否需要数据增强
from keras.preprocessing.image import ImageDataGenerator
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:   这将进行预处理和实时数据扩充
    '''
    Keras的图像生成器ImageDataGenerator。这个生成器有很多操作如翻转、旋转和缩放等，目的是生成更加多
    且不一样的图像数据，这样我们得到的训练模型泛化性更加的好，从而得到的模型更加准确。
    '''
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset  设置数据集上的输入均值为0
        samplewise_center=False,  # set each sample mean to 0              设置每个样本均值为0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset  将输入除以数据集的标准差
        samplewise_std_normalization=False,  # divide each input by its std 将每个输入除以它的标准差
        zca_whitening=False,  # apply ZCA whitening  使用ZCA白化图像
        zca_epsilon=1e-06,  # epsilon for ZCA whitening  为ZCA白化
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180) 在范围(深度，0到180)内随机旋转图像
        # randomly shift images horizontally (fraction of total width)
        # 水平随机移动图像(总宽度的一部分)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        # 垂直随机移动图像(总高度的一部分)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear  设定随机剪切范围
        zoom_range=0.,  # set range for random zoom  设置范围为随机变焦
        channel_shift_range=0.,  # set range for random channel shifts  设置范围的随机通道移位
        # set mode for filling points outside the input boundaries  设置输入边界外的填充点模式
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"   用于fill_mode的值= "常量"
        horizontal_flip=True,  # 随机翻转图片
        vertical_flip=False,   # 随机翻转图片
        # 设置重新调平因子(在任何其他转换之前应用)
        rescale=None,
        # 设置将应用于每个输入的函数
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        # 图像数据格式，可以是“channels_first”，也可以是“channels_last”
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        # 保留用于验证的图像部分(严格在0到1之间)
       # validation_split=0.0
        )

    #  计算特征标准化所需的数量
    # (std, mean, and principal components if ZCA whitening is applied).(标准差，平均值，如果使用ZCA美白的主要成分)。
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().将模型安装到datagen.flow()生成的批上。
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )

#--------------------------------------------------------------------------------------------------------------------#
#  ------------------------------------------- 保存模型 -------------------------------------------------------------#
# 打印模型框架
print(model.summary())
# 保存模型
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#---------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ 训练过程可视化 -----------------------------------------------------#

import matplotlib.pyplot as plt
# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])    # 训练准确率
plt.plot(history.history['val_acc'])# 测试准确率
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss']) # 训练损失
plt.plot(history.history['val_loss'])# 测试损失
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
#---------------------------------------------------------------------------------------------------------------------#

# Total params: 14,848,586
# Trainable params: 7,213,322
# Non-trainable params: 7,635,264
# Epoch 1/1
# 1875/1875 [==============================] - 229s 122ms/step - loss: 0.7084 - acc: 0.7634 - val_loss: 0.6659 - val_acc: 0.8142
