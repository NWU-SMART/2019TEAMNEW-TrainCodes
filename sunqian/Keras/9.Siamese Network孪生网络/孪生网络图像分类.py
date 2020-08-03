# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/8/3
# 文件名称：孪生网络-图像分类.py
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入数据、数据预处理
# 3、构建模型
# 4、训练模型
# 5、训练可视化
# 6、模型预测
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
import gzip
import numpy as np
import random
from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D, Input, Lambda
from keras import regularizers
from keras.models import Model
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
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    with gzip.open(paths[2],'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(),np.uint8,offset=8)

    with gzip.open(paths[3],'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(),np.uint8,offset=16).reshape(len(y_test),28,28)
    return (x_train, y_train),(x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()
# 查看训练集和测试集的总数
print('train_images:', x_train.shape, x_train.dtype)
print('train_labels:', y_train.shape, y_train.dtype)
print('test_images:', x_test.shape, x_test.dtype)
print('test_labels:', y_test.shape, x_test.dtype)

# -------输出结果------
# train_images: (60000, 28, 28) uint8
# train_labels: (60000,) uint8
# test_images: (10000, 28, 28) uint8
# test_labels: (10000,) uint8
# -------输出结果------

# 查看训练集的第一张图片
plt.figure()
plt.imshow(x_train[0,:])
plt.colorbar()
plt.grid(False)
plt.show()

# 数据归一化
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /=255.
x_test /=255.

# 数据切片
# 分割标签["top","trouser","pullover","coat","sandal","ankle boat"]
digit_indices = [np.where(y_train == i)[0] for i in {0, 1, 2, 4, 5, 9}]
digit_indices = np.array(digit_indices)

# 求每列的长度
n = min([len(digit_indices[d]) for d in range(6)])

# 所选的6个标签对应的图片中80%训练 20%测试
train_set_shape = n * 0.8
test_set_shape = n * 0.2
y_train_new = digit_indices[:, :int(train_set_shape)]
y_test_new = digit_indices[:, int(train_set_shape):]

# 剩下的4个标签["dress","sneaker","bag","shirt"]对应的图片全部用来测试
digit_indices_t = [np.where(y_train == i)[0] for i in {3, 6, 7, 8}]
y_test_new_2 = np.array(digit_indices_t)

print(y_train_new.shape)
print(y_test_new.shape)
print(y_test_new_2.shape)

# -------输出结果------
# (6, 4800)
# (6, 1200)
# (4, 6000)
# -------输出结果------

# 创建图片对
# 为了建立一个能够识别两个图像是否属于同一类的分类器，我们需要在整个数据集中创建一对又一对的图像
#  1）对于属于每个给定类的每个图像，选择它旁边的图像并形成一对。例如在“top”类中，第一图像和第二图像将形成一对，第二图像将与第三图像形成一对，这些对是正对（positive pairs）
#  2） 同时选择一个属于另一个类的图像并形成一对。例如“top”类中的第一个图像将与“pullover”类中的第一个图像形成一对。这些对是负对（negative pairs）
#  3） 我们将正负对的每个组合的标签指定为[1,0]

def create_pairs( x, digit_indices):
    pairs = []
    labels = []
    class_num = digit_indices.shape[0]
    for d in range(class_num):
        for i in range(int(digit_indices.shape[1]) - 1):
            # 创建正对
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            pairs += [[x[z1], x[z2]]]
            # 创建负对
            inc = random.randrange(1, class_num)
            dn = (d+inc) % class_num
            z1, z2 = digit_indices[d][i],digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

# 训练对
tr_pairs, tr_y = create_pairs(x_train, y_train_new)
tr_pairs = tr_pairs.reshape(tr_pairs.shape[0], 2, 28, 28, 1)
print(tr_pairs.shape)

# 测试对1
te_pairs_1, te_y_1 = create_pairs(x_train, y_test_new)
te_pairs_1 = te_pairs_1.reshape(te_pairs_1.shape[0], 2, 28, 28, 1)
print(te_pairs_1.shape)

# 测试对2
te_pairs_2, te_y_2 = create_pairs(x_train, y_test_new_2)
te_pairs_2 = te_pairs_2.reshape(te_pairs_2.shape[0], 2, 28, 28, 1)
print(te_pairs_2.shape)
# -------输出结果------
# (57588, 2, 28, 28, 1)
# (14388, 2, 28, 28, 1)
# (47992, 2, 28, 28, 1)
# -------输出结果------
#  -------------------------- 2、导入数据、数据预处理-----------------------------------

#  -------------------------- 3、构建模型 -------------------------------------------
def create_base_network(input_shape):
    # 共享权重
    input = Input(shape=input_shape)
    x = Conv2D(32, (7, 7), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01),
               bias_regularizer=regularizers.l1(0.01))(input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l1(0.01))(x)
    return Model(input, x)

input_shape = (28, 28, 1)
base_network = create_base_network(input_shape)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
# 因为我们重复使用同一个实例“base_network”，所以网络的权重将在两个分支之间共享
processed_a = base_network(input_a)
processed_b = base_network(input_b)
print(base_network.summary())

# 添加一个lambda层
from keras import backend as K
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model([input_a, input_b], distance)
#  -------------------------- 3、构建模型 ------------------------------------------

#  -------------------------- 4、训练模型------------------------------------------
from keras.optimizers import RMSprop
rms= RMSprop()
def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy'])
result = model.fit([tr_pairs[:, 0],tr_pairs[:, 1]], tr_y,
                   batch_size=128,
                   epochs=10,
                   validation_data=([te_pairs_1[:, 0],te_pairs_1[:, 1]],te_y_1))

#  -------------------------- 4、训练模型------------------------------------------

#  -------------------------- 5、训练可视化-------------------------------------------
# 绘制训练和验证的准确率值
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.savefig('Siamese_Network_valid_acc.png')
plt.show()

# 绘制训练和验证的损失值
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Valid'],loc='upper left')
plt.savefig('Siamese_Network_valid_loss.png')
plt.show()
#  -------------------------- 5、训练可视化 ------------------------------------------

#  -------------------------- 6、模型预测-------------------------------------------
y_pred = model.predict([tr_pairs[:,  0], tr_pairs[:, 1]])

def compute_accuracy(y_true, y_pred):  # numpy上的操作
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def  accuracy(y_true, y_pred): # Tensor上的操作
    #  用固定的距离阈值计算分类精度
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtpe)))

tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = model.predict([te_pairs_1[:, 0], te_pairs_1[:, 1]])
te_acc = compute_accuracy(te_y_1, y_pred)
print(' * Accuracy on training set:%0.2f%%' %(100 * tr_acc))
print(' * Accuracy on test set:%0.2f%%' %(100 * te_acc))
# 结果
# * Accuracy on training set:94.64%
# * Accuracy on test set:94.20%
#  -------------------------- 6、模型预测-------------------------------------------
