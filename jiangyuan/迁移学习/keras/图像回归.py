# ----------------开发者信息--------------------------------
# 开发者：姜媛
# 开发日期：2020年6月22日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------


# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 8、训练过程可视化
# ----------------------   代码布局： ----------------------


#  -------------------------- 1、导入需要包 -------------------------------
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 -------------------------------


#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------
# 本地读取数据
path = 'C:\\Users\\HP\\Desktop\\每周代码学习\\迁移学习\\数据集'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train = f['x_train']
# 测试数据
X_test = f['x_']
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
# --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、伪造回归数据 ---------------------
# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']
# 给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]  # 均值列表


def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]  # 均值mean,标准差std,数量
    return np.round(price, 2)


y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))
#  --------------------- 3、伪造回归数据 ---------------------


#  --------------------- 4、数据归一化 ---------------------
# y_train_price_pd = y_train_pd['price'].tolist()
# y_test_price_pd = y_test_pd['price'].tolist()
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))
#  --------------------- 4、数据归一化 ---------------------


#  --------------------- 5、迁移学习建模 ---------------------
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
#  --------------------- 5、迁移学习建模 ---------------------


#  --------------------- 6、训练 ---------------------
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
    # ImageDataGenerator对图像进行预处理，对图像的处理集合warp起来作为一个整体使用
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 使输入数据集去中心化（均值为0），按feature执行
        samplewise_center=False,  # 使输入数据的每个样本均值为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差来完成标准化，按feature执行
        samplewise_std_normalization=False,  # 将输入的每个样本除以标准差
        zca_whitening=False,  # 应用ZCA白化
        # （假设训练数据是图像，由于图像中相邻像素之间具有很强的相关性，所以用于训练时输入是冗余的。白化的目的就是降低输入的冗余性。）
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # 整数，数据提升时图片随机转动的角度。随机选择图片的角度，是取值为0~180的度数
        width_shift_range=0.1,  # 浮点数，图片宽度的某个比例，数据提升时图片随机水平偏移的幅度
        height_shift_range=0.1,  # 浮点数，图片高度的某个比例，数据提升时图片随机竖直偏移的幅度。
        shear_range=0.,  # 浮点数，剪切强度（逆时针方向的剪切变换角度）。是用来进行剪切变换的程度
        zoom_range=0.,  # 用来进行随机的放大
        channel_shift_range=0.,  # 浮点数，随机通道偏移的幅度
        fill_mode='nearest',  # ‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
        cval=0.,  # 浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
        horizontal_flip=True,  # 布尔值，进行随机水平翻转。随机的对图片进行水平翻转，这个参数适用于水平翻转不影响图片语义的时候
        vertical_flip=False,  # 进行随机竖直翻转
        rescale=None,  # 将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，
        # 所以我们将这个值定为0~1之间的数。
        preprocessing_function=None,
        # 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），
        # 并且输出一个具有相同shape的numpy array
        data_format=None,
        # 字符串，channels_last(默认)或channels_first之一，表示输入中维度的顺序。
        # channels_last对应输入尺寸为(batch, height, width, channels)，
        # channels_first对应输入尺寸为(batch, channels, height, width)。
        # 它默认为从Keras配置文件~ /.keras / keras.json中找到的image_data_format值。
        # 如果你从未设置它，将使用"channels_last"。
        validation_split=0.0  # fraction of images reserved for validation (strictly between 0 and 1)
    )

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    print(x_train.shape[0] // batch_size)  # 取整
    print(x_train.shape[0] / batch_size)  # 保留小数
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),  # 按batch_size大小从x,y生成增强数据
                                  # flow_from_directory()从路径生成增强数据,和flow方法相比最大的优点在于不用
                                  # 一次将所有的数据读入内存当中,这样减小内存压力，这样不会发生OOM
                                  epochs=epochs,
                                  steps_per_epoch=x_train.shape[0] // batch_size,
                                  validation_data=(x_test, y_test),
                                  workers=10  # 在使用基于进程的线程时，最多需要启动的进程数量。
                                  )
#  --------------------- 6、训练 ---------------------


#  --------------------- 7、模型可视化与保存模型 ---------------------
model.summary()
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
#  --------------------- 7、模型可视化与保存模型 ---------------------


#  --------------------- 8、训练过程可视化 ---------------------
# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#  --------------------- 8、训练过程可视化 ---------------------
