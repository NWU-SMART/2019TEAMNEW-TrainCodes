# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.05
# 开发框架：keras
# 温馨提示：服务器上跑
#----------------------------------------------------------------------------------------------------------------------#

# -------------------------------------------------代码布局------------------------------------------------------------#
# 1、加载手写体数据集  图像数据预处理
# 2、伪造回归数据
# 3、建立模型
# 4、保存模型与模型可视化
# 5、训练过程可视化
#----------------------------------------------------------------------------------------------------------------------#
#-----------------------------------------加载手写体数据集  图像数据预处理---------------------------------------------#
import numpy as np
# 载入数据本地：
# path = 'F:\\Keras代码学习\\keras\\keras_datasets\\mnist.npz'
# 载入数据服务器：
path = 'mnist.npz'
f = np.load(path)
print(f.files) # 查看文件内容 ['x_test', 'x_train', 'y_train', 'y_test']
# 定义训练数据 60000个
x_train = f['x_train']
# 定义训练标签
y_train = f['y_train']
# 定义测试数据 10000个
x_test = f['x_test']
# 定义测试标签
y_test = f['y_test']
f.close()
# 打印训练数据的维度
print(x_train.shape)  # (60000, 28, 28)
# 打印测试数据的维度
print(x_test.shape)   # (10000, 28, 28)

# 数据预处理
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
# 将数据变为array数组类型
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
# 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.

#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------伪造回归数据----------------------------------------------------------#
# 转换成DataFrame格式方便数据处理（DataFrame是一个表格型的数据类型，每列值类型可以不同，是最常用的pandas对象。）
import pandas as pd
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

# 把0-9数字当做十类衣服，为其设置价格
mean_value_list = [45,57,85,99,125,27,180,152,225,33] # 均值列表
'''
这是的np是numpy包的缩写，np.random.normal()的意思是一个正态分布，normal这里是正态的意思。
numpy.random.normal(loc=0,scale=1e-2,size=shape) ，意义如下： 
参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
参数size(int 或者整数元组)：输出的值赋在shape里，默认为None。
'''
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)],3,size=1))[0] # 均值mean,标准差std,数量shape
    return np.round(price,2) # 返回按指定位数进行四舍五入的数值(这里保留两位)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)
print(y_train_pd.head())           # 打印前五个训练标签
print('--------------------')
print(y_test_pd.head())            # 打印前五个测试标签
#    label   price
# 0      5   23.24
# 1      0   40.96
# 2      4  123.72
# 3      1   58.55
# 4      9   32.76

#----------------------------------------------------------------------------------------------------------------------#
# ----------------------------------------MinMaxScaler数据归一化-------------------------------------------------------#
# MinMaxScaler：归一到 [ 0，1 ] ；MaxAbsScaler：归一到 [ -1，1 ]
from sklearn.preprocessing import MinMaxScaler

# 训练标签归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:,1]


# 测试集标签归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:,1]

y_test_label = min_max_scaler.transform(y_test_pd)[:,0]# 归一化后的标签

print(len(y_train))  # 60000
print(len(y_test))   # 10000
#----------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------建立迁移学习模型--------------------------------------------------------#
# 使用vgg16模型
from keras import applications
base_model = applications.VGG16(include_top = False,weights = 'imagenet',input_shape = x_train.shape[1:]) # 第一层需要指出图像的大小
print(x_train.shape[1:])# 1875
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
model.add(Dense(1))# 输出层是1类
model.add(Activation('linear'))
# 合并vgg16和自己的两层全连接
model = Model(input=base_model.input,outputs= model(base_model.output))

# 保持搭建的网络的前15层（共15层）权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False

#----------------------------------------------------------------------------------------------------------------------#
batch_size = 32            # 设置批大小为32
epochs = 5
data_augmentation = True   # 使用图像增强

import os
save_dir = os.path.join(os.getcwd(), 'saved_models_keras_transferlearning') # 保存的模型路径
model_name = 'keras_fashion_transferlearning_trained_model.h5'            # 模型名字


import keras
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6) # 使用RMSprop优化器

model.compile(loss='mse',
              optimizer=opt,                   # 使用之前定义的优化器
              metrics=['accuracy'])
#----------------------------------------------------数据增强----------------------------------------------------------#
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

#----------------------------------------------------------------------------------------------------------------------#
#  ------------------------------------------- 保存模型 ---------------------------------------------------------------#
# 打印模型框架
print(model.summary())
# 保存模型
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#----------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------ 训练过程可视化 ------------------------------------------------------#

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
#----------------------------------------------------------------------------------------------------------------------#

# 实验结果：
# Epoch 1/5
# 1875/1875 [==============================] - 252s 134ms/step - loss: 0.0359 - acc: 3.3333e-05 - val_loss: 0.0088 - val_acc: 2.0000e-0
# 4
# Epoch 2/5
# 1875/1875 [==============================] - 251s 134ms/step - loss: 0.0113 - acc: 3.3333e-05 - val_loss: 0.0092 - val_acc: 2.0000e-0
# 4
# Epoch 3/5
# 1875/1875 [==============================] - 251s 134ms/step - loss: 0.0087 - acc: 3.3333e-05 - val_loss: 0.0091 - val_acc: 2.0000e-0
# 4
# Epoch 4/5
# 1875/1875 [==============================] - 251s 134ms/step - loss: 0.0076 - acc: 3.3333e-05 - val_loss: 0.0040 - val_acc: 2.0000e-0
# 4
# Epoch 5/5