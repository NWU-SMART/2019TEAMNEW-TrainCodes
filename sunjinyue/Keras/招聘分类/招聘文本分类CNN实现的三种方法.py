#-------------------------------开发者信息----------------------------------
#开发人：孙进越
#日期：2020.6.3
#开发内容：招聘推荐的CNN的三种实现方法
#
#
#


#--------------------------------导入包-------------------------------------
import pandas as pd #数据处理包
import jieba #中文词库包
import jieba.analyse as analyse
import numpy as np
import matplotlib.pyplot as plt #画图包
from sklearn.model_selection import train_test_split #数据划分包
from sklearn.preprocessing import LabelEncoder  #数据有类别编码
# 导入需要的keras包
from keras.preprocessing.text import Tokenizer  #将文本处理成索引类型的数据
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization

#---------------------------------招聘数据导入--------------------------------

#数据路径
path = 'D:\\应用软件\\研究生学习\\job_detail_dataset.csv'     # 对比读取 npz数据方法   path = ''
#读取数据
data = pd.read_csv(path,encoding='utf-8')                    #  data = np.load(path)

print(data.head(5))     # 前五个
print(data.shape)       # 50000*2

#-------------------------------数据预处理----------------------

label = list(data['PositionType'].unique())
print(label)

# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row) #返回label列表对应值的索引
    return num_label

#给不同工作类型上数字标签  【0，项目管理】
data['label'] = data['PositionType'].apply(label_dataset)
#丢弃缺失部分
data = data.dropna()
print(data.head(5))

# 分词
def chinese_word(row):
    return" ".join(jieba.cut(row))
data['chinses_cut'] = data.Job_Description.apply(chinese_word)

#提取关键词
# analyse.extract_tags(texts,topK,withWeight,allowPOS)
'''
第一个参数：待提取关键词的文本
第二个参数：返回关键词的数量，重要性从高到低排序
第三个参数：是否同时返回每个关键词的权重
第四个参数：词性过滤，为空表示不过滤，若提供则仅返回符合词性要求的关键词
'''
def key_word_extract(texts):
  return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
data['keyword']=data.Job_Description.apply(key_word_extract)
data.head(5)

#建立字典
token = Tokenizer(num_words=1000)
#按照单词出现的顺序建立
token.fit_on_texts(data['keyword'])
#将文本转化为单词序列
description = token.texts_to_sequences(data['keyword'])
#将序列填充为最大的50个
job_description = sequence.pad_sequences(description,maxlen=50)

# 训练集
x_train = job_description
y_train = data['label'].tolist()   # ????????
#print(y_train.head(5))  不可这么写
#y_train = np.array(y_train)
#print(y_train)

#-------------------------数据预处理---------------------

#-------------------------Sequential---------------------
#
# model = Sequential()
# model.add(Embedding(input_dim=2000, #字典大小？？？？？？
#                     output_dim=32,  # 嵌入维度
#                     input_length=50 #数字列表长度？？？？？？
#                     )
#           )
# model.add(Conv1D(filters=256,
#                  kernel_size=3,
#                  padding='same',
#                  activation='relu'
#                  )
#           )
# model.add(MaxPool1D(pool_size=3,
#                     strides=3,
#                     padding='same'
#                     )
#           )
# model.add(Conv1D(filters=32,
#                  kernel_size=3,
#                  padding='same',
#                  activation='relu'
#                  )
#           )
# model.add(Flatten())
# model.add(Dropout(0.3))
# model.add(BatchNormalization())  # (批)规范化层
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(units=10,
#                 activation="softmax"
#                 )
#           )
# --------------------API-----------------------------
# from keras.layers import Input
# from keras.models import Model
# inputs = Input(shape=(50,))
# x = Embedding(output_dim=32, input_dim=2000)(inputs)
# x = Conv1D(256, 3, activation='relu')(x)  # 训练256个卷积核，每个卷积核大小3*3
# x = MaxPool1D()(x)
# x = Conv1D(32, 3, activation='relu')(x)
# x = Flatten()(x)
# x = Dropout(0.2)(x)
# x = BatchNormalization()(x)  # 解决数据分布不平衡
# x = Dense(units=256, activation='relu')(x)
# x = Dropout(0.2)(x)
# outputs = Dense(units=10, activation="softmax")(x)
# model = Model(inputs=inputs, outputs=outputs)

# ------------------ class--------------------*/
from keras.layers import Input
import keras
inputs = Input(shape=(50,))
class Cnnmodel(keras.Model):                  #为什么这个模型进不去？？？？？？？？？？？
    def __init__(self):
        super(Cnnmodel, self).__init__()
        self.embedding = Embedding(output_dim=32, input_dim=1000)
        self.conv1 = Conv1D(256, 3, activation='relu')
        self.maxpool = MaxPool1D()
        self.conv2 = Conv1D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.dropout1 = Dropout(0.3)
        self.BatchNormal = BatchNormalization()
        self.dense1 = Dense(units=256, activation='relu')
        self.dropout2 = Dropout(0.25)
        self.dense2 = Dense(units=10, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.BatchNormal(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


model = Cnnmodel()

#  -------------------------- 模型编译和训练------------------------------
batch_size = 256
epochs = 5


#model.summary()  # 可视化模型
model.compile(loss='sparse_categorical_crossentropy',  # 多分类
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=2
    # 训练集的20%用作验证集
)

# --------------------------保存模型，显示运行结果----------------------------
from keras.utils import  plot_model

#保存模型
model.save('model_Cnn_Text.h5')  # 模型文件名称
#模型可视化
plot_model(model,to_file='model_Cnn_Text.png',show_shapes=True)

from keras.models import load_model

#加载模型

print(x_train[0])
y_new = model.predict(x_train[0].reshape(1,50))
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])

import  matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
