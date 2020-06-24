# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/23
# 文件名称：电影评论分类
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------
# ----------------------   代码布局： ----------------------
# 1、导入需要的包
# 2、导入路透社数据集
# 3、构建网络
# 4、验证使用的方法
# 5、模型可视化
# 6、从头开始训练一个新模型
# 7、模型预测
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要的包 -------------------------------
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要的包 -------------------------------

#  -------------------------- 2、导入路透社数据集 -------------------------------
#数据存在本地路径E:\\keras_datasets\\reuters.npz
path='E:\\keras_datasets\\reuters.npz'
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(path=path,num_words=10000)
#8982个训练样本和2246个测试样本

#将数据向量化
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)

#处理多分类问题的标签有两种方法：
# 1.通过分类编码（one-hot编码）对标签进行编码，然后使用categorical_crossentropy作为损失函数
#2.将标签编码为整数，然后使用sparse_categorical_crossentropy损失函数
#本次使用one-hot编码将标签向量化，one-hot编码就是将每个标签标示为全零向量，只有标签索引对应的元素为1
def to_one_hot(labels,dimension=46):
    results=np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1.
    return results
#将训练标签和测试标签向量化
one_hot_train_labels=to_categorical(train_labels)
one_hot_test_labels=to_categorical(test_labels)


#  -------------------------- 2、导入路透社数据集-------------------------------

#  -------------------------- 3、构建网络 -------------------------------
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000, )))
model.add(layers.Dense(64,activation='relu'))
#对于单标签，多分类问题，网络的最后一层应该使用softmax激活，这样可以输出在N个输出类别上的概率分布
model.add(layers.Dense(46,activation='softmax'))
#编译模型，使用分类交叉熵损失函数categorical_crossentropy
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
#  -------------------------- 3、构建网络-------------------------------

#  -------------------------- 4、验证使用的方法-------------------------------
#在训练数据中留出1000个样本作为验证集
x_val=x_train[:1000]
partial_x_train=x_train[1000:]
y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]
#训练模型，共20轮次
history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=512,
                  validation_data=(x_val,y_val))
#  -------------------------- 4、验证使用的方法-------------------------------


#  -------------------------- 5、模型可视化-------------------------------
#绘制训练损失和验证损失
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#绘制训练精度和验证精度
plt.clf()#清空图像
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training acc')#'bo'表示蓝色圆点
plt.plot(epochs,val_acc,'b',label='Validation acc')#‘b'表示蓝色实线
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#  -------------------------- 5、模型可视化 -------------------------------

#  -------------------------- 6、从头开始训练一个新网络 -------------------------------
#网络在训练9轮后开始过拟合，从头开始训练一个新网络，共9个轮次，然后在测试集上评估模型
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000, )))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
#编译模型，使用分类交叉熵损失函数categorical_crossentropy
model.compile(optimizer='rmsprop',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.fit(partial_x_train,
                  partial_y_train,
                  epochs=9,
                  batch_size=512,
                  validation_data=(x_val,y_val))
results=model.evaluate(x_test,one_hot_test_labels)
results

import copy
test_labels_copy=copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array=np.array(test_labels)==np.array(test_labels_copy)
float(np.sum(hits_array))/len(test_labels)
#  -------------------------- 6、从头开始训练一个新网络-------------------------------

#-----------------------------7、模型预测------------------------------------------
predictions=model.predict(x_test)
#predictions[0]中每个元素都是长度为46的向量
#predictions[0].shape
#这个向量的所有元素总和为1
#np.sum(predictions[0])
#最大的元素就是预测类别，即概率最大的类别(是4)
#np.argmax(predictions[0])
#-----------------------------7、模型预测----------------------------------------
