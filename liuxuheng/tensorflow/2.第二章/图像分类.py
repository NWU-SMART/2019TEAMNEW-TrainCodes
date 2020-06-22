# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年6月19日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
sess = tf.InteractiveSession()
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、数据载入与预处理 -------------------------------
def load_data():
    paths = [
        'D:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'D:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]
    # 加载数据返回4个NumPy数组
    with gzip.open(paths[0], 'rb') as lbpath: # 读压缩文件
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        # frombuffer将data以流的形式读入转化成ndarray对象
        # 第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    return (x_train, y_train)
(x_train, y_train)= load_data()# 载入数据
x_train = x_train.astype('float32')#数据类型转换
x_train /= 255  #归一化

x_train = tf.reshape(x_train, [-1, 784]) #(60000,784)
x_train = tf.Session().run(x_train) # 转为ndarray数组

y_train = LabelBinarizer().fit_transform(y_train)# 对y进行one-hot编码
y_train =np.array(y_train)
#  -------------------------- 2、数据载入与预处理 -------------------------------

#  -------------------------- 3、搭建模型 -------------------------------
# 产生随机变量，符合 normal 分布
# 传递 shape 就可以返回weight和bias的变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 定义2维的 convolutional 层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 定义 pooling 图层
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 定义placeholder 
x = tf.placeholder(tf.float32, [None, 784])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x, [-1, 28, 28, 1])# (60000,28,28,1)


W_conv1 = weight_variable([3, 3, 1, 32]) # 3x3卷积，输入通道1，输出通道32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出大小28,28,32
h_pool1 = max_pool_2x2(h_conv1) # 输出大小14,14,32


W_conv2 = weight_variable([3, 3, 32, 64])# 3x3卷积，输入通道32，输出通道64              
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输出大小14,14,64
h_pool2 = max_pool_2x2(h_conv2) # 输出大小7,7,64


W_fc1 = weight_variable([7*7*64, 1024]) # 全连接层，变为1024
b_fc1 = bias_variable([1024]) 
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # 将h_pool2的结果平展
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # 经过全连接层
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


W_fc2 = weight_variable([1024, 10]) # 全连接层，变为10
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 经过全连接层

y_ = tf.placeholder(tf.float32, [None, 10])  #定义placeholder 
#  -------------------------- 3、搭建模型 -------------------------------

#  -------------------------- 4、训练模型   --------------------------------
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(prediction), reduction_indices=[1])) # 交叉熵损失函数
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)# 选择自适应优化器Adagrad，把学习速率设为0.3

sess = tf.Session()
#初始化变量
init=tf.global_variables_initializer()
sess.run(init)
for i in range(10):
    sess.run(train_step, feed_dict={x: x_train, y_: y_train, keep_prob: 0.5})
    print(y_)
#  -------------------------- 4、训练模型   --------------------------------
