#------------------------------------------------开发者信息-------------------------------------------
#开发人：王园园
#开发日期：2020.6.2
#开发软件：pycharm
#开发项目：图像分类：迁移学习（keras）

#-------------------------------------------------代码布局---------------------------------------------
#1、导包
#2、读取数据及与图像预处理
#3、迁移学习建模
#4、训练
#5、模型可视化与保存模型
#6、训练过程可视化

#---------------------------------------------------导包----------------------------------------------
import gzip
import os
import cv2
import numpy as np
from keras import applications, Sequential, Model
from keras.layers import Flatten, Dense, Activation
from keras_preprocessing.image import ImageDataGenerator
from networkx.drawing.tests.test_pylab import plt
from tensorflow import keras

#--------------------------------------------------读取数据及图像预处理----------------------------------
path = 'D:/keras_datasets/'
def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]
    #训练数据标签
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #训练数据
    with gzip.open(path[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    #测试数据标签
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    #测试数据
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'keras_fashion_transfer_learning_trained_model.h5'  #保存模型为.h5文件
#将训练与测试的标签转换成独热编码
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#由于mist的输入数据维度是（num， 28， 28）， vgg16需要三维图像，因为扩充一下mnist的最后一维
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_DRAY2RGB) for i in x_test]
#数组
x_train = np.asarray(X_train)
x_test = np.asarray(x_test)
#数据类型是float32
x_train = x_train.astype('float32')
x_test = x_test.astype(('float32'))
#数据归一化
x_train /= 255
x_test /= 255

#----------------------------------------------------迁移学习建模------------------------------------------
#使用VGG16模型（include_top=False表示，不包含最后的3个全连接层）
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
print(x_train.shape[1:])
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
#7*7*512---->256
model.add(Dense(num_classes))
model.add(Activation('softmax'))
#输出为VGG16的数据，经过VGG16的特征层，2层全连接到num_classes输出
model = Model(inputs=base_model.input, outputs=model(base_model.output))
#冻结VGG16的前15层，权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False
#优化器
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#编译模型
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#----------------------------------------------------------训练-----------------------------------------------
#如果没有使用数据增强
if not data_augmentation:
    print('Not using data augmentation')
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
else:
    # 进行数据增强
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,             #在数据集上设置输入均值为0
        samplewise_center=False,              #设置每个样本均值为0
        featurewise_std_normalization=False,  #根据数据集的std划分输入
        samplewise_std_normalization=False,  #将每个输入除以它的std
        zca_whitening=False,                  #应用ZCA美白
        zca_epsilon=1e-06,                    #用于ZCA增白
        rotation_range=0,                     #在范围内随机旋转图像(角度，0到180)
        width_shift_range=0.1,                #水平随机移动图像(总宽度的分数)
        height_shift_range=0.1,               #垂直随机移动图像(总高度的一部分)
        shear_range=0.,                       #设置随机剪切的范围
        zoom_range=0.,                        #设置范围为随机变焦
        channel_shift_range=0.,               #设置范围随机通道移位
        fill_mode='nearest',                  #设置输入边界外的填充点模式
        cval=0.,
        horizontal_flip=True,                 #随机水平翻转图片
        vertical_flip=False,                  #随机垂直翻转图片
        rescale=None,                         #设置重新调平因子(在任何其他转换之前应用)
        preprocessing_function=None,          #设置将应用于每个输入的函数
        data_format=None,
        validation_split=0.0)                 #保留用于验证的图像的比例(严格在0和1之间)
datagen.fit(x_train)
print(x_train.shape[0]//batch_size)
print(x_train.shape[0]/batch_size)
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              epochs=epochs,
                              steps_per_epoch=x_train.shape[0]//batch_size,
                              validation_data=(x_test, y_test),
                              workers=10)

#------------------------------------------------------保存模型---------------------------------------
model.summary()
#判断路径是否存在，不存在创建
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)  #保存模型

#------------------------------------------------------训练过程可视化-----------------------------------
#绘制训练与验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

#绘制训练与验证的损失
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()





