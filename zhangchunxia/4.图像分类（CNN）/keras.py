# ----------------开发者信息-----------------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月4日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息------------------------------------------------
# ----------------------   代码布局： ----------------------
# 1、导入需要的的包
# 2、数据读取
# 3、数据预处理
# 4、建立模型
# 5、训练模型
# 6、保存模型及显示结果
# ----------------------   代码布局： ----------------------
#  -------------------------- 1、导入需要包 -------------------------------
from keras import Input, Model
from tensorflow.keras.utils import get_file
import gzip
import numpy as np
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import functools
from tensorflow.keras import optimizers
import os        #由于图像数据太大，需要显卡加速训练
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用第3块显卡
#  -------------------------- 1、导入需要包 ------------------------------
#  -------------------------- 2、读取数据---------------------------------
def load_data():
    path = ['D:/northwest/小组视频/4图像分类/train-labels-idx1-ubyte.gz','D:/northwest/小组视频/4图像分类/train-images-idx3-ubyte.gz'
           'D:/northwest/小组视频/4图像分类/t10k-labels-idx1-ubyte.gz','D:/northwest/小组视频/4图像分类/t10k-images-idx3-ubyte.gz'
            ]
    with gzip.open(path[0],'rb') as lpath:
        y_train = np.frombuffer(lpath.read(),np.unit8,offset=8)
    with gzip.open(path[1],'rb') as ipath:
        x_train = np.frombuffer(ipath.read(),np.unit8,offset=16).reshape(len(y_train),28,28,1)
    with gzip.open(path[2],'rb') as lpath:
        y_text = np.frombuffer(ipath.read(),np.unit8,offset=8)
    with gzip.open(path[3],'rb') as ipath:
        x_text = np.frombuffer(ipath.read(),np.unit8,offset=16).reshape(len(y_text),28,28,1)
    return(y_train,x_train),(y_text,x_text)
(y_train,x_train),(y_text,x_text) = load_data()
#  -------------------------- 2、读取数据---------------------------------
#  -------------------------- 3、数据预处理-------------------------------
#对标签信息进行独热编码，共有10类标签
y_train = keras.utils.to_categorical(y_train,10)
x_train = keras.utils.to_categorical(x_train,10)
#对图像信息转换为数据类型
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_train /= 255
x_text /=255
#  -------------------------- 3、数据预处理-------------------------------
#  -------------------------- 4、建立模型---------------------------------
#/----------------------------method1-API方法-----------------------------
inputs = Input(shape=x_train.shape[1:])
x = Conv2D(32,(3,3),padding='same',activation='relu')(inputs)
x = Conv2D(32,(3,3),activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64,(3,3),padding='same',activation='relu')(x)
x = Conv2D(64,(3,3),activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10,activation='softmax')(x)
model = Model(input=inputs,output=x)
#/----------------------------method1-API方法-----------------------------
#/----------------------------method2-Sequential方法----------------------
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu'))#第一层需要指出输入图像的大小
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(activation='relu')
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(activation='softmax')
#/----------------------------method2-Sequential方法----------------------
# /----------------------------method3-class方法--------------------------
class TX(keras.Model):
    def __init__(self, kears=None):
        super(TX,self).__init(name='TX')
        self.conv2d1 = keras.layer.Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:],activation='relu')
        self.conv2d2 = kears.layer.Conv2D(32,(3,3),activation='relu')
        self.maxpooling2d = kears.layer.MaxPooling2D(pool_size=(2, 2))
        self.dropout = keras.layer.Dropout(0.25)

        self.conv2d1 = keras.layer.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2d2 = kears.layer.Conv2D(64, (3, 3), activation='relu')
        self.maxpooling2d = kears.layer.MaxPooling2D(pool_size=(2, 2))
        self.dropout = keras.layer.Dropout(0.25)

        self.flatten = keras.layer.Flatten()
        self.dense1 = kears.layer.Dense(512,activation='relu')
        self.dropout = keras.layer.Dropout(0.25)
        self.dense2 = keras.layer.Dense(10,activation='softmax')
        self.dropout = keras.layer.Dropout(0.5)
#/----------------------------method3-class方法--------------------------
optimize = keras.optimizers.rmsprop(lr=1e-4,decay=1e-6)
model.compile(loss='categorical_crossentropy',optimizer=optimize,metrics=['accuracy'])
#  -------------------------- 4、建立模型---------------------------------
#  -------------------------- 5、训练模型---------------------------------
# 通过判断是否使用图片增强技术去训练图片
data_augmentation = True
if not data_augmentation:
    print('没有图像增强技术')
    history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_text,y_text), shuffle=True)#随机打乱样本顺序
else:
    print('使用了图像增强技术')
    data = ImageDataGenerator(featurewise_center=False,#输入数据的均值设置为0
                              samplewise_center=False,#输入样本的均值设置为0
                              featurewise_std_normalization=False,#将输入除以数据集的标准差
                              samplewise_std_normalization=False,#每个输入除以标准差
                              zca_whitening=False,#是否使用zca白化（降低输入的冗余度），zca白化主要用于去相关性，让白化后的数据更接近于原始数据
                              zca_epsilon=1e-06,#利用阈值构建低通滤波器，使其对输入数据进行过滤
                              rotation_range=0,#随机旋转的度数
                              width_shift_range=0.1,#水平随机移动的宽度0.1
                              height_shift_range=0.1,#垂直随机移动的宽度0.1
                              shear_range=0,#不随机裁剪
                              channel_shift_range=0,#通道不随机转换
                              fill_mode='nearnest',#靠近哪个店就用哪个点填充
                              cval=0,#边界之外点的值
                              horizontal_flip=True,#随机水平翻转
                              vertical_flip=False,#不随机水平翻转
                              rescale=None,#不进行缩放（否则数据乘以所提供的值）
                              preprocessing_function=None,#应用于输入的函数
                              data_format=None,#输入图像格式
                              validation_split=0.0,#用于验证图像的比例
    )
data.fit(x_train)
print(x_train.shape[0]//32)#取整
print(x_train.shape[0]/32)#保留小数
history = model.fit_generator(data.flow(x_train,y_train,batch_size=32,),#按batch_size大小将x,y生成增强数据
                              epochs=5,steps_per_epoch=x_train.shape[0]//32,#每批次训练的样本
                              validation_data=(x_text,y_text), works=10)#最大进程数
#  -------------------------- 5、训练模型---------------------------------
#  -------------------------- 6、显示结果 --------------------------------
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train','Test'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train','Test'],loc='upper left')#设置图例位置左上角
plt.show()
#  -------------------------- 6、显示结果 --------------------------------

