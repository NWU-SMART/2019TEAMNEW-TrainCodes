# ----------------开发者信息----------------------------
# 开发者：张涵毓
# 开发日期：2020年5月28日
# 内容：招聘信息文本分类-CLASS
# 修改内容：
# 修改者：
# ----------------开发者信息----------------------------

# ----------------------代码布局-------------------------------------
# 1.引入keras，matplotlib，numpy，sklearn，pandas,jieba包
# 2.导入招聘信息数据
# 3.分词和提取关键词
# 4.建立和使用字典
# 5.CLASS
# 6.显示结果
# ---------------------------------------------------------------------

#-------------------------1、导入相关包-------------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
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
#-------------------------1、导入相关包-------------------------------------

#-------------------------2、导入招聘信息数据----------------------------------
#读取数据路径  中文编码
job_detail_pd = pd.read_csv('D:\keras_data\job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))  #显示前5个数据
label = list(job_detail_pd['PositionType'].unique())  # 标签
print(label)


# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label

job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)
#-------------------------2、导入招聘信息数据----------------------------------

#-------------------------3、分词和提取关键词------------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
#参数：待提取关键词的文本；返回关键词的数量 ；是否返回每个关键词的权重；词性过滤，为空表示不过滤，若提供则仅返回符合词性要求的关键词

job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
#  -------------------------- 3、分词和提取关键词 -------------------------------

#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)

x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist() #生成列表，但和list不同
#  -------------------------- 4、建立字典，并使用 -------------------------------
#------------------------------5、CLASS---------------------------
batch_size = 256
epochs = 5

import keras
from keras import Model,Input

inputs=Input(shape=(50,))

class MLPic(keras.Model):
    def __init__(self):
        super(MLPic, self).__init__()
        self.embedding =Embedding(output_dim=32,input_dim=2000)
        self.dropout1 = Dropout(0.2)
        self.flatten = Flatten()
        self.dense1= Dense(units=256,activation='relu')
        self.dropout2=Dropout(0.25)
        self.dense2=Dense(units=10,activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

model = MLPic()
model.compile(loss='sparse_categorical_crossentropy',  # keras的多类损失函数
                                                       #都是计算多分类crossentropy的，只是对y的格式要求不同。
                                                       #1）如果是categorical_crossentropy，那y必须是one-hot处理过的
                                                       #2）如果是sparse_categorical_crossentropy，那y就是原始的整数形式，数字编码，比如[1, 0, 2, 0, 2]这种
              optimizer="adam",
              metrics=["accuracy"]
              )
result = model.fit(x_train, y_train, batch_size=256, epochs=5, verbose=2,validation_split=0.2)


#------------------------------5、CLASS---------------------------

#  -------------------------- 6、显示运行结果 -------------------------------

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
