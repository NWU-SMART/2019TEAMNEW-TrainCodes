# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年5月28日
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization


#   ---------------------- 数据的载入和处理 ----------------------------
job_detail_pd = pd.read_csv('job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))
label = list(job_detail_pd['PositionType'].unique())  # 标签
print(label)


# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)


#  -------------------------- 分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))


job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)

#  -------------------------- 建立字典 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50  词嵌入前的预处理
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)  # 长度都填充到50
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()  # 把数组转化为列表


#  -------------------------- 构建模型的三种方法-------------------------------
batch_size = 256
epochs = 5
# ------------------------Class继承------------------
from keras.layers import Input
from keras.models import Model

inputs = Input(shape=(50,))


class JobModel(keras.Model):  # 继承keras.Model
    def __init__(self):   # 绑定属性
        super(JobModel, self).__init__()
        self.embedding = Embedding(output_dim=32, input_dim=2000)
        self.dropout = Dropout(0.2)
        self.flatten = Flatten()
        self.dense1 = Dense(units=256, activation='relu')
        self.dense2 = Dense(units=10, activation='softmax')

    def call(self, inputs):  # 模型调用的代码
        x = self.embedding(inputs)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x



model = JobModel()



model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])


history = model.fit(x_train, y_train, batch_size=256, epochs=5, verbose=2, validation_split=0.2)


# / /------------------ 模型训练--------------------*/


# / /------------------ 结果显示--------------------*/
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# / /------------------ 结果显示--------------------*/