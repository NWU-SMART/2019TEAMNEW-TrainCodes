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
# 修改内容：将keras三种定义方法写入 /将原先第三种方法存在的错误改正
# /------------------ 开发者信息 --------------------*/

# /------------------ 代码布局 --------------------*/
'''
代码布局
1.导入需要的包
2.读取数据
3.数据预处理
4.建立模型
5.模型训练
'''
# /------------------ 代码布局 --------------------*/


# /------------------ 导入需要的包 --------------------*/
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
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization

# /------------------ 导入需要的包 --------------------*/

# /------------------ 读取数据 --------------------*/

# 读取数据
# 数据路径
path = 'G:\python\code\多层感知器\job_detail_dataset.csv'
# path = 'job_detail_dataset.csv'
# 读取数据
data = pd.read_csv(path,encoding='utf-8')
# 维度（50000 x 2 ）
# print(data)

# /------------------ 读取数据 --------------------*/


# /------------------ 数据预处理 --------------------*/
# 数据处理
#去重读取工作类型 10类
# ['项目管理', '移动开发', '后端开发', '前端开发', '测试', '高端技术职位', '硬件开发', 'dba', '运维', '企业软件']
label = list(data['PositionType'].unique())

# 为工作描述设置标签的id
def label_dataset(row):
     num_label = label.index(row)  # 返回label列表对应值的索引
     return num_label

# 给不同的工作类型打上分类标签
# ['0',项目管理']这种
data['label'] = data['PositionType'].apply(label_dataset)
# 将中间数据缺失的部分丢弃 50000*2 -> 44831*3
data = data.dropna()

# 提取描述中的中文分词并写入
# 采用的精确模式  他来到上海交通大学  ->   他/ 来到/ 上海交通大学
# (若参数cut_all=True)  ->  他/ 来到/ 上海/ 上海交通大学/ 交通/ 大学
def chinese_word(row):
    return " ".join(jieba.cut(row))
data['chinese_cut'] = data.Job_Description.apply(chinese_word)

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
# 将文本装换为单词序列
description = token.texts_to_sequences(data['keyword'])
# 将序列填充到最大的50个
job_description = sequence.pad_sequences(description,maxlen=50)

# 选取训练集
x_train = job_description
y_train = data['label'].tolist()
y_train = np.array(y_train)
print(x_train.shape)
print()

# /------------------ 数据预处理 --------------------*/

# /------------------ 模型建立--------------------*/

batch_size = 256
epochs = 5
# /------------------ 序贯模型--------------------*/
# model = Sequential()
# model.add(Embedding(output_dim=32,  # 词向量的维度
#                     input_dim=1000,  # 字典大小
#                     input_length=50  # 每个数字列表的长度
#                     ))
#
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(units=256,
#                 activation="relu"))
# model.add(Dropout(0.25))
# model.add(Dense(units=10,
#                 activation="softmax"))
# /------------------ 序贯模型（第二种类型）--------------------*/
# model = Sequential([
#     Embedding(output_dim=32,  # 词向量的维度
#               input_dim=1000,  # 字典大小
#               input_length=50),
#     Dropout(0.2),
#     Flatten(),
#     Dense(units=256,
#           activation='relu'),
#     Dropout(0.25),
#     Dense(units=10,
#           activation='softmax')
# ])
# /------------------ API类型--------------------*/
#
# from keras.layers import Input
# from keras.models import Model
# input = Input(shape=(50,))
# train_API = Embedding(output_dim=32,input_dim=1000)(input)
# train_API = Dropout(0.2)(train_API)
# train_API = Flatten()(train_API)
# train_API = Dense(units=256,activation='relu')(train_API)
# train_API = Dropout(0.25)(train_API)
# result = Dense(units=10,activation='softmax')(train_API)
# model = Model(input=input,outputs=result)

# 类继承形式(没有办法保存模型，只能保存训练的模型的参数)
import keras
class MLP(keras.Model):
    def __init__(self):
        super(MLP,self).__init__(name='mlp')
        self.embedding = keras.layers.Embedding(output_dim=32,
                                                input_dim=1000,input_length=50)
        self.dense1 = keras.layers.Dense(units=256,activation='relu')
        self.dense2 = keras.layers.Dense(units=10,activation='softmax')
        self.dropout1 = keras.layers.Dropout(0.2)
        self.dropout2 = keras.layers.Dropout(0.25)
        self.flatten = keras.layers.Flatten()

    def call(self,x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

# /------------------ 模型建立--------------------*/


# /------------------ 模型训练保存--------------------*/
model = MLP()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
result = model.fit(x_train,y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=2,
                   validation_split=0.2)

model.save('mlp.h5')
model_new = load_model('mlp.h5')
y_predict = model_new.predict(x_train[0].reshape(1,50))

# /------------------ 模型训练保存--------------------*/


# /------------------ 结果显示--------------------*/
# 损失显示
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss-epoch')
plt.show()
# 准确率显示
plt.plot(result.history['accuracy'])
plt.plot(result.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('accuracy-epoch')
plt.show()
