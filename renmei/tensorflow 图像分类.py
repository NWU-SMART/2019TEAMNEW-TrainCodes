#---------------------------------开发者信息----------------------------------------
#姓名：任梅
#日期：2020.05.25
#---------------------------------开发者信息----------------------------------------

#--------------------------------代码布局———————————————————-
#1.导入需要包
#2.导入数据
#3.构建函数
#4构建模型
#5.训练和评估

#--------------------------------导入需要包————————————————————-
import gzip
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#--------------------------------导入需要包————————————————————-

#--------------------------------导入数据———————————————————-
def load_data():
    paths = [
        'D:\\keras_datasets\\train-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\train-images-idx3-ubyte.gz',
        'D:\\keras_datasets\\t10k-labels-idx1-ubyte.gz', 'D:\\keras_datasets\\t10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
#--------------------------------导入数据———————————————————-

(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 32
num_classes = 10
epochs = 5
lr=0.001
dropout1=0.25
dropout2=0.5
x=tf.placeholder(tf.float32,shape=[None,784],name="imput")
y=tf.placeholder(tf.float32,shape=[None,num_classes],name="output")

#--------------------------构建函数——————————————————————————————
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def conv2d2(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
x_image = tf.reshape(x, [-1,28,28,1])
#--------------------------构建函数——————————————————————————————



#=================构建模型--------------------------------------------------------
if __name__ == '__main__':

# 第一层卷积层
     W_conv1 = weight_variable([3, 3, 1, 32])
     b_conv1 = bias_variable([32])
     h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
     h_pool1 = max_pool_2x2(h_conv1)

#第二层
     W_conv2 = weight_variable([3, 3, 32, 32])
     b_conv2 = bias_variable([32])
     h_conv2 = tf.nn.relu(conv2d2(x_image, W_conv2) + b_conv2)
     h_pool2 = max_pool_2x2(h_conv2)
     h_pool2_drop = tf.nn.dropout(h_pool2,dropout1)#防止过拟合

#第三层
     W_conv3 = weight_variable([3, 3, 32, 64])
     b_conv3 = bias_variable([64])
     h_conv3 = tf.nn.relu(conv2d(x_image, W_conv3) + b_conv3)
     h_pool3 = max_pool_2x2(h_conv3)

#第四层
     W_conv4 = weight_variable([3, 3, 32, 64])
     b_conv4 = bias_variable([64])
     h_conv4 = tf.nn.relu(conv2d2(x_image, W_conv4) + b_conv4)
     h_pool4= max_pool_2x2(h_conv4)
     h_pool4_drop = tf.nn.dropout(h_pool4,dropout1)#防止过拟合

#全连接
     W_fc1 = weight_variable([24 * 24* 64, 512])
     b_fc1 = bias_variable([512])
     h_pool2_flat = tf.reshape(h_pool4_drop, [-1, 24 * 24 * 64])
     h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
     keep_prob = tf.placeholder(tf.float32)
     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#第二个全连接
     W_fc2 = weight_variable([512, 10])
     b_fc2= bias_variable([10])
     y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#=================构建模型--------------------------------------------------------

#-----------------------------训练和评估--------------------------------
     loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
# 同样定义train_step
     train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # 定义测试的准确率
     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     sess = tf.InteractiveSession()
     sess.run(tf.global_variables_initializer())
     for i in range(20000):
        sess.run(train_step, feed_dict={x: batch_size, y: batch_size, keep_prob: 0.5})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch_size[0], y: batch_size[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch_size[0], y: batch_size[1], keep_prob: 0.5})

# 训练结束后报告在测试集上的准确度
        print("test accuracy %g" % accuracy.eval(feed_dict={
                     x: x_test, y: y_test, keep_prob: 1.0}))













