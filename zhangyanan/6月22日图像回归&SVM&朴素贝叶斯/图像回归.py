# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月22日
# 修改日期：
# 修改人：
# 修改内容：

#  -------------------------- 导入需要包 -------------------------------
import keras
import numpy as np
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import applications
from keras.models import Sequential,Model
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
import cv2
import matplotlib.pyplot as plt

plt.style.use('ggplot') # 画的更好看

#  --------------------- 读取手写体数据及与图像预处理 ---------------------
batch_size = 32
epochs = 5
data_augmentation = True

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 我电脑之前下载过mnist 这里用load_data()直接载入
# 观察下X_train和X_test维度
print(x_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(x_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 由于mist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,因为扩充一下mnist的最后一维
x_train = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_train]
x_test = [cv2.cvtColor(cv2.resize(i,(48,48)),cv2.COLOR_GRAY2RGB)for i in x_test]
# 将数据变为array数组类型,否则后面会报错
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
# /255 归一化
x_train = x_train.astype("float32")/255.
x_test  = x_test.astype("float32")/255.

print('X_train shape:', x_train.shape)  # (60000, 48, 48, 3)
print(x_train.shape[0], 'train samples')  # 60000 train samples
print(x_test.shape[0], 'test samples')   # 10000 test samples

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

#  --------------------- 伪造回归数据 ---------------------

# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

# 给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]  # 均值列表
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3,size=1))[0] #均值mean,标准差std,数量
    return np.round(price, 2)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

#  --------------------- 数据归一化 ---------------------
# y_train_price_pd = y_train_pd['price'].tolist()
# y_test_price_pd = y_test_pd['price'].tolist()
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max_scaler.fit(y_test_pd)  # 我感觉去掉这一行精度会提高， 原因说不太清楚
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))


#  --------------------- 迁移学习建模 ---------------------

# 使用VGG16模型
base_model = applications.VGG16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])  # 第一层需要指出图像的大小

# # path to the model weights files.
# top_model_weights_path = 'bottleneck_fc_model.h5'
print(x_train.shape[1:])
model = Sequential()
print(base_model.output)
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('linear'))

# add the model on top of the convolutional base
model = Model(inputs=base_model.input, outputs=model(base_model.output))  # VGG16模型与自己构建的模型合并

# 保持VGG16的前15层权值不变，即在训练过程中不训练
for layer in model.layers[:15]:
    layer.trainable = False

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='mse',
              optimizer=opt,
              )


#  --------------------- 训练 ---------------------

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 使输入数据集去中心化（均值为0）, 按feature执行
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差以完成标准化, 按feature执行
        samplewise_std_normalization=False,  # 将输入的每个样本除以其自身的标准差。
        zca_whitening=False,  # 对输入数据施加ZCA白化
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # 整数，数据提升时图片随机转动的角度。随机选择图片的角度，是一个0~180的度数，取值为0~180
        width_shift_range=0.1,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
        height_shift_range=0.1,  # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。
        shear_range=0.,  # 浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度
        zoom_range=0.,
        # 浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]。用来进行随机的放大
        channel_shift_range=0.,  # 浮点数，随机通道偏移的幅度
        fill_mode='nearest',  # 浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        cval=0.,  # 布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
        horizontal_flip=True,  # 布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
        vertical_flip=False,  # 进行随机竖直翻转
        rescale=None,  # 将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
        preprocessing_function=None,
        # 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
        data_format=None,
        #  “channel_first”或“channel_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channel_last”
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print(x_train.shape[0]//batch_size)  # 取整
    print(x_train.shape[0]/batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train,  # 按batch_size大小从x,y生成增强数据
                                     batch_size=batch_size),
                        # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                        # 一次将所有的数据读入内存当中,这样减小内存压，这样不会发生OOM
                        epochs=epochs,
                        steps_per_epoch=x_train.shape[0]//batch_size,
                        validation_data=(x_test, y_test),
                        workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                       )

model.summary()

#  --------------------- 训练过程可视化 ---------------------
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

