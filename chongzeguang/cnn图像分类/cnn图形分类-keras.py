# ----------------开发者信息--------------------------------#
# 开发者：崇泽光
# 开发日期：2020年6月8日
# 修改日期：
# 修改人：
# 修改内容：
# --------------------------------------------------------#
# 导入需要包

import gzip
import numpy as np # NumPy 是一个运行速度非常快的数学库，主要用于数组计算
import keras
from keras.preprocessing.image import ImageDataGenerator # ageDataGenerator()是keras.preprocessing.image模块中的图片生成器，同时也可以在batch中对数据进行增强，扩充数据集大小，增强模型的泛化能力。比如进行旋转，变形，归一化等等。
from keras.models import Sequential # keras中的主要数据结构是model（模型），它提供定义完整计算图的方法。通过将图层添加到现有模型/计算图，我们可以构建出复杂的神经网络。Sequential模型可以构建非常复杂的神经网络，包括全连接神经网络、卷积神经网络(CNN)、循环神经网络(RNN)
from keras.layers import Dense, Dropout, Activation, Flatten # Dense层(全连接层），Dropout层用于防止过拟合，激活层对一个层的输出施加激活函数，Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
from keras.layers import Conv2D, MaxPooling2D #二维卷积层即对图像的空域卷积，该层对二维输入进行滑动窗卷积；为空域信号施加最大值池化
import os # 操作系统接口模块

# 读取数据与数据预处理
def load_data():
    paths = [
        'D:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'D:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8) #numpy.frombuffer 用于实现动态数组，返回数组的数据类型，offset是读取的起始位置，默认为0。

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1) #图片尺寸28*28

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()

#确定超参数
batch_size = 32 #一次训练所选取的样本数
num_classes = 10 #一共有10种类型
epochs = 5 #定义为向前和向后传播中所有批次的单次训练迭代。简单说，epochs指的就是训练过程中数据将被“轮”多少次
data_augmentation = True  # 图像增强，数据增强主要用来防止过拟合，用于dataset较小的时候。
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn') #路径拼接os.path.join()函数；使用os.getcwd()函数获得当前的路径。
model_name = 'keras_fashion_trained_model.h5'

y_train = keras.utils.to_categorical(y_train, num_classes) #将标签转换为分类的 one-hot 编码，y为int数组，num_classes为标签类别数
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32') #输入数据类型转换，下边进行数值归一化
x_test = x_test.astype('float32')

x_train /= 255  # 归一化；除法赋值运算符
x_test /= 255  # 归一化

#搭建CNN模型
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',  # 32，(3,3)是卷积核数量和大小
                 input_shape=x_train.shape[1:]))  # 第一层需要指出图像的大小
model.add(Activation('relu')) #激活层
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #最大池化
model.add(Dropout(0.25)) #Dropout层用于防止过拟合，训练时概率性丢弃

model.add(Conv2D(64, (3, 3), padding='same')) #如果padding设置为same，说明输入图片大小和输出图片大小是一致的
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小
model.add(Dense(512)) #为model添加Dense层，即全链接层，512为输出
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes)) # 再次添加Dense层，10维输出
model.add(Activation('softmax')) #通过softmax激励函数进行分类

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) #优化器

#模型训练设置
model.compile(loss='categorical_crossentropy', #交叉熵损失函数
              optimizer=opt, #优化器
              metrics=['accuracy'])

#训练
if not data_augmentation: #图像增强
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train, #模型拟合
              batch_size=batch_size, #一次训练所选取的样本数
              epochs=epochs, #训练过程中数据将被“轮”多少次
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.') #预处理和实时数据扩充
    datagen = ImageDataGenerator(
        featurewise_center=False,  #使输入数据集去中心化（均值为0）
        samplewise_center=False,  #使输入数据的每个样本均值为0。
        featurewise_std_normalization=False, #将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization=False,  #布尔值，将输入的每个样本除以其自身的标准差。
        zca_whitening=False,  #否应用 ZCA 白化。
        zca_epsilon=1e-06,  # ZCA 白化的 epsilon 值，默认为 1e-6。
        rotation_range=0,  # 整数。随机旋转的度数范围。
        width_shift_range=0.1, #浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度。
        height_shift_range=0.1, #浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。
        shear_range=0.,  #浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度。
        zoom_range=0.,  #浮点数或形如[lower,upper]的列表，随机缩放的幅度，用来进行随机的放大
        channel_shift_range=0.,  #浮点数，随机通道偏移的幅度。
        fill_mode='nearest', #当进行变换时超出边界的点将根据本参数给定的方法进行处理
        cval=0.,  #浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值。
        horizontal_flip=True,  #布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候。
        vertical_flip=False,  #布尔值，进行随机竖直翻转。
        rescale=None, #值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
        preprocessing_function=None, #将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片，（秩为3的numpy array），并且输出一个具有相同shape的numpy array
        data_format=None, #代表图像的通道维的位置
        validation_split=0.0) #浮点数。Float. 保留用于验证的图像的比例（严格在0和1之间）。

    # 计算按特征进行归一化所需的数量
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # 将模型拟合到由datagen.flow（）生成的批次上。
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size, #表示是将一个epoch分成多少个batch_size，
                        validation_data=(x_test, y_test), #验证数据
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )

#保存模型
model.summary() #通过model.summary()输出模型各层的参数状况
if not os.path.isdir(save_dir): #os.path.isdir()用于判断对象是否为一个目录。
    os.makedirs(save_dir) #递归目录创建函数
model_path = os.path.join(save_dir, model_name) #os.path.join()函数：连接两个或更多的路径名组件
model.save(model_path) #保存模型
print('Saved trained model at %s ' % model_path)

#显示运行结果
import matplotlib.pyplot as plt
# 绘制训练集和验证集的准确率值
plt.plot(history.history['accuracy']) #调用plot函数在当前的绘图对象中绘图
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy') #设置图表的标题
plt.ylabel('Accuracy') #设置y轴的文字
plt.xlabel('Epoch') #设置x轴的文字
plt.legend(['Train', 'Valid'], loc='upper left') #显示label中标记的图示
plt.savefig('tradition_cnn_valid_acc.png') #保存生成的图片
plt.show() #显示出所有的绘图对象

# 绘制训练集和验证集的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
