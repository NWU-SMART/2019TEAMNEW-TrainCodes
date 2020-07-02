# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/7/2
# 文件名称：图像分类.py
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
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import os
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入数据、数据预处理 -------------------------------------------
def load_data():
    paths = [
        'E:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'E:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'E:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'E:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0],'rb') as lbpath: #'rb'指读取二进制文件，非人工书写的数据如.jpg等
        y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)

    with gzip.open(paths[1],'rb') as imgpath:
        # frombuffer将data以流的形式读入转化成ndarray对象
        # 第一参数为stream，第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train),28,28,1)
    with gzip.open(paths[2],'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(),np.uint8,offset=8)

    with gzip.open(paths[3],'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(),np.uint8,offset=16).reshape(len(y_test),,28,28,1)
    return (x_train, y_train),(x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()

batch_size = 32
num_classes = 10
epochs = 10
data_augmentation = True # 图像增强
num_prediction = 20
# os.getcwd()返回当前进程的工作目录
save_dir = os.path.join(os.getcwd(),'saved_models_cnn')  #os.path.join()函数连接两个或更多的路径名组件
model_name = 'keras_fashion_trained_model.h5'

# 将类别转换成独热编码
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化
x_train /=255
x_test /=255
#  -------------------------- 2、导入数据、数据预处理--------------------------------------------

#  -------------------------- 3、构建模型 -------------------------------------------
model = Sequential()
model.add(Conv2D(32, (3, 3), # 32,(3,3)是卷积核数量和大小
                 padding='same',
                 input_shape=x_train.shape[1:]))
# 第一层需要指出图像的大小
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 初始化RMSprop优化器
opt = keras.optimizers.RMSprop(lr=0.0001,decay=1e-6) #lr指学习率learning rate decay指学习率每次更新的下降率
# 使用RMSprop优化器
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.summary()
#  -------------------------- 3、构建模型 ------------------------------------------

#  -------------------------- 4、训练模型------------------------------------------
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
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0,
        zoom_range=0,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0)

    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size) #取整
    print(x_train.shape[0]/batch_size) #保留小数
    # 拟合模型
    # 按batch_size大小从x,y生成增强数据
    # flow_from_directory()从路径生成增强数据，和flow方法相比最大的优点在于不用一次性将所有数据读入内存中，可以减小内存压力
    history = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test,y_test),
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
plt.legend(['Train','Valid'],loc='upper lift')
plt.savefig('tradition_cnn_valid_acc.png')
plt.show()

# 绘制训练和验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper lift')
plt.savefig('tradition_cnn_valid_loss.png')
plt.show()
# 训练精度可达到85%，验证精度可达到87%
#  -------------------------- 5、训练可视化 ------------------------------------------
