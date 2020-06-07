# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.5.20
#
# 版本号：Versoin 1.0
#
# 修改日期：2020.5.27
#
# 修改人：于林生
#
# 修改内容：写入keras的三种形式
# / /------------------ 开发者信息 --------------------*/


# / /------------------ 导入需要的包 --------------------*/
# 导入需要的包
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
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D,Input
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization

# / /------------------ 导入需要的包 --------------------*/


# / /------------------ 读取数据 --------------------*/
# 读取数据
# 数据路径
path = 'G:/python/code/多层感知器/pytorch改编代码/job_detail_dataset.csv'
# 读取数据
data = pd.read_csv(path,encoding='utf-8')
# 维度（50000 x 2 ）
print(data)#PositionType Job_Description

# / /------------------ 导入需要的包 --------------------*/

# / /------------------ 数据处理 --------------------*/

# 数据处理
label = list(data['PositionType'].unique())#去重读取工作类型 10类
print(label)
print(label.index('项目管理'))#找到项目管理的索引（list的属性）
# 为工作描述设置标签的id
def label_dataset(row):
     num_label = label.index(row)  # 返回label列表对应值的索引
     return num_label

# 给不同的工作类型打上分类标签
data['label'] = data['PositionType'].apply(label_dataset)
print(data)
data = data.dropna()
print(data)#44831*3

# 提取描述中的中文分词并写入
# 采用的精确模式  他来到上海交通大学  ->   他/ 来到/ 上海交通大学
# (若参数cut_all=True)  ->  他/ 来到/ 上海/ 上海交通大学/ 交通/ 大学
def chinese_word(row):
    return " ".join(jieba.cut(row))
data['chinese_cut'] = data.Job_Description.apply(chinese_word)
data.head(5)


# 提取关键词
# 提取关键词
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

# 建立字典
token = Tokenizer(num_words=1000)
# 按照单词出现的顺序建立
token.fit_on_texts(data['keyword'])

description = token.texts_to_sequences(data['keyword'])
job_description = sequence.pad_sequences(description,maxlen=50)
# 选取训练集
x_train = job_description
y_train = data['label'].tolist()
y_train = np.array(y_train)#类继承方式需要将y_train转换类型，其他的不需要
print(x_train)
print(y_train)

# / /------------------ 数据处理--------------------*/


# / /------------------ 模型定义--------------------*/
# cnn模型
        ## / /------------------ 序贯模型--------------------*/
# model =  Sequential()
# model.add(Embedding(1000,32,input_length=50))
# model.add(Conv1D(256,3,padding='same',activation='relu'))
# model.add(MaxPool1D(3,3,padding='same'))
# model.add(Conv1D(32,3,padding='same',activation='relu'))
# model.add(Flatten())
# model.add(Dropout(0.3))
# model.add(BatchNormalization())
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(units=10,activation='softmax'))
## / /------------------ 序贯模型第二种--------------------*/

# model = Sequential([
#     Embedding(1000,32,input_length=50),
#     Conv1D(256,3,padding='same',activation='relu'),
#     MaxPool1D(3,3,padding='same'),
#     Conv1D(32,3,padding='same',activation='relu'),
#     Flatten(),
#     Dropout(0.3),
#     BatchNormalization(),
#     Dense(256,activation='relu'),
#     Dropout(0.3),
#     Dense(units=10,activation='softmax')
# ])
## / /------------------ API模式--------------------*/
# from keras import Model
# input = Input(shape=(50,))
# x_train1 = Embedding(1000,32,input_length=50)(input)
# x_train1 = Conv1D(256,3,padding='same',activation='relu')(x_train1)
# x_train1 = MaxPool1D(3,3,padding='same')(x_train1)
# x_train1 = Conv1D(2,3,padding='same',activation='relu')(x_train1)
# x_train1 = Flatten()(x_train1)
# x_train1 = Dropout(0.3)(x_train1)
# x_train1 = BatchNormalization()(x_train1)
# x_train1 = Dense(256,activation='relu')(x_train1)
# x_train1 = Dropout(0.3)(x_train1)
# result = Dense(units=10,activation='softmax')(x_train1)
# model = Model(inputs=input,outputs=result)

## / /------------------ 类继承的方式--------------------*/
import keras
class cnn(keras.Model):
    def __init__(self):
        super(cnn,self).__init__(name='cnn')
        self.embedding = keras.layers.Embedding(1000,32,input_length=50)
        self.conv1 = keras.layers.Conv1D(256,3,padding='same',activation='relu')
        self.maxpool = keras.layers.MaxPool1D(3,3,padding='same')
        self.conv2 = keras.layers.Conv1D(2,3,padding='same',activation='relu')
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(0.3)
        self.batchnonrmal = keras.layers.BatchNormalization()
        self.dense1 = keras.layers.Dense(256,activation='relu')
        self.dense2 = keras.layers.Dense(units=10,activation='softmax')
    def call(self,x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.batchnonrmal(x)
        x = self.dense1(x)
        x = self.dropout(x)
        result = self.dense2(x)
        return result




# / /------------------ 模型定义--------------------*/


# / /------------------ 模型训练--------------------*/
# 模型训练
model = cnn()
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
result_cnn = model.fit(x_train,y_train,batch_size=256,epochs=5,verbose=2,validation_split=0.2)


# / /------------------ 模型训练--------------------*/


# / /------------------ 结果显示--------------------*/
import matplotlib.pyplot as plt
plt.plot(result_cnn.history['loss'])
plt.plot(result_cnn.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.plot(result_cnn.history['accuracy'])
plt.plot(result_cnn.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# / /------------------ 结果显示--------------------*/