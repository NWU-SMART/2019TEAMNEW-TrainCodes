# ----------------开发者信息--------------------------------#
# 开发人员：孙迁
# 开发日期：2020/6/22
# 文件名称：电影评论分类
# 开发工具：PyCharm
# ----------------开发者信息--------------------------------#
#  -------------------------- 1、导入需要包 -------------------------------
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、准备数据 -----------------------------
#加载IMDB数据集
#(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000) 国外服务器无法访问

# 数据放到本地路径
# E:\\keras_datasets\\imdb.npz(本地路径)
path = 'E:\\keras_datasets\\imdb.npz'
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(path=path,num_words=10000)
#数据放到本地路径
#将整数序列编码为二进制矩阵
def vectorize_sequences(sequences,dimension=10000):
    #创建形状为(len(sequence,dimension))的零矩阵
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        #将result[i]的指定索引设为1
        results[i,sequence]=1.
    return results
#将训练数据和测试数据向量化
x_train=vectorize_sequences(train_data)
x_test=vectorize_sequences((test_data))
#将标签向量化
y_train=np.asarray(train_labels).astype('float32')
y_test=np.asarray(test_labels).astype('float32')
#  -------------------------- 2、准备数据 -----------------------------


#  -------------------------- 3、构建网络 -----------------------------
model=models.Sequential()
#两个中间层，每层都有16个隐藏单元。使用relu作为激活函数
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
#第三层输出一个标量，预测当前评论的情感。使用sigmoid激活以输出一个0~1范围内的概率值
model.add(layers.Dense(1,activation='sigmoid'))
#使用rmsprop优化器和二元交叉熵损失函数配置模型
#model.compile(optimizer='rmsprop',
 #             loss='binary_crossentropy',
  #            metrics=['accuracy'])
#配置优化器
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
#使用自定义的损失和指标
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])
#  -------------------------- 3、构建网络 -----------------------------


#  -------------------------- 4、验证使用的方法 -----------------------------
#为了在训练过程中监控模型在前所未见的数据上的精度，需要将原始训练数据留出10000个样本作为验证集
x_val=x_train[:10000]
partial_x_train=x_train[10000:]
y_val=y_train[:10000]
partial_y_train=y_train[10000:]
#训练模型。使用512个样本组成的小批量，将模型训练20个轮次，同时监控留出的10000个样本上的损失和精度
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history=model.fit(partial_x_train,
                  partial_y_train,
                  epochs=20,
                  batch_size=512,
                  validation_data=(x_val,y_val))
#  -------------------------- 4、验证使用的方法 -----------------------------

#  -------------------------- 5、模型可视化 -----------------------------
history_dict=history.history
loss_values=history_dict['loss']
val_loss_values=history_dict['val_loss']
epochs=range(1,len(loss_values)+1)

plt.plot(epochs,loss_values,'bo',label='Training loss') #'bo'表示蓝色圆点
plt.plot(epochs,val_loss_values,'b',label='Validation loss')#'b'表示蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#  -------------------------- 5、模型可视化 -----------------------------

# ---------------------------6.模型预测----------------------------------
model.predict(x_test)
#----------------------------6.模型预测-----------------------------------