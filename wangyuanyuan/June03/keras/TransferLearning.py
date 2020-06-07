#---------------------------------------------------开发者信息----------------------------------------------
#开发人：王园园
#开发日期：2020.6.03
#开发软件：pycharm
#开发项目：图像回归：迁移学习（keras）

#----------------------------------------------------代码布局-----------------------------------------------
#1、导包
#2、读取手写体数据及图像预处理
#3、构建自编码器模型
#4、模型可视化
#5、训练
#6、查看自编码器的压缩效果
#7、查看自编码器的解码效果
#8、训练过程可视化

#---------------------------------------------------导包---------------------------------------------------
import os
import numpy as np
import pandas as pd
from keras import applications, Sequential, Model, optimizers
from keras.layers import Flatten, Dense, Dropout, Activation
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler

#------------------------------------------------------读取手写提数据及图像预处理-----------------------------
path = 'D:\keras_datasets\mnist.npz'
f = np.load(path)
x_train = f['x_train']    #训练数据
y_train = f['y_train']    #训练数据标签
x_test = f['x_test']       #测试数据
y_test = f['y_test']       #测试数据标签
f.close()
#数据标准化
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#np.prod是将28*28矩阵转化成1*784，方便BP神经网络输入层784个神经元读取
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

#伪造回归数据
#转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
#设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']
#给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 33]  #均值列表
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]  #均值mean，标准差std，数量
    return np.round(price, 2)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

#数据归一化
#训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train_label = min_max_scaler.transform(y_train_pd)[:, 1]
#验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]

#------------------------------------------------------------------迁移学习建模--------------------------------------------
#使用VGG16模型
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
print(x_train.shape[1:])
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
model = Model(inputs=base_model.input, outputs=model(base_model.output))
#冻结VGG16的前15层权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False
#优化器
opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)
#模型编译
model.compile(loss='mse', optimizer=opt)

#----------------------------------------------------------------------模型训练---------------------------------------------
#如果没有数据增强
data_augmentation = True
batch_size = 32
epochs = 5
if not data_augmentation:
    print('Not using data augmentation')
    history = model.fit(x_train, y_train_label, batch_size=batch_size, epochs=epochs,
                        validation_data=(x_test, y_test_label), shuffle=True)
else:
    #进行数据增强
    print('Using real_time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 在数据集上设置输入均值为0
        samplewise_center=False,  # 设置每个样本均值为0
        featurewise_std_normalization=False,  # 根据数据集的std划分输入
        samplewise_std_normalization=False,  # 将每个输入除以它的std
        zca_whitening=False,  # 应用ZCA美白
        zca_epsilon=1e-06,  # 用于ZCA增白
        rotation_range=0,  # 在范围内随机旋转图像(角度，0到180)
        width_shift_range=0.1,  # 水平随机移动图像(总宽度的分数)
        height_shift_range=0.1,  # 垂直随机移动图像(总高度的一部分)
        shear_range=0.,  # 设置随机剪切的范围
        zoom_range=0.,  # 设置范围为随机变焦
        channel_shift_range=0.,  # 设置范围随机通道移位
        fill_mode='nearest',  # 设置输入边界外的填充点模式
        cval=0.,
        horizontal_flip=True,  # 随机水平翻转图片
        vertical_flip=False,  # 随机垂直翻转图片
        rescale=None,  # 设置重新调平因子(在任何其他转换之前应用)
        preprocessing_function=None,  # 设置将应用于每个输入的函数
        data_format=None,
        validation_split=0.0)  # 保留用于验证的图像的比例(严格在0和1之间)

    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train_label, batch_size=batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0]//batch_size,
                                  validation_data=(x_test, y_test_label),
                                  workers=10)

#--------------------------------------------------模型可视化并保存模型---------------------------------------
model.summary()
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'keras_fashion_transfer_learning_trained_model.h5'
#如果路径不存在。创建该路径
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)   #保存模型

#----------------------------------------------------------训练过程可视化--------------------------------------
import matplotlib.pyplot as plt
#绘制训练与验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
