# ----------------开发者信息--------------------------------#
# 开发者：姜媛
# 开发日期：2020年5月28日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#


# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
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

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 文件放置在目录  D:\keras_datasets\job_detail_dataset.csv
job_detail_pd = pd.read_csv('C:\\Users\\HP\\Desktop\\每周代码学习\\招聘信息文本分类\\job_detail_dataset.csv', encoding='UTF-8')
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


#  -------------------------- 2、招聘数据数据导入 -------------------------------

#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))


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
y_train = job_detail_pd['label'].tolist()
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  -------------------------- 5、class类继承模型 -------------------------------
batch_size = 256
epochs = 5
inputs = Input(shape=(50,))

class employment(keras.Model):  # 定义employment
    def __init__(self):  # 定义初始化
        super(employment, self).__init__()
        self.embedding = keras.layers.Embedding(2000, 32)
        self.dropout1 = keras.layers.Dropout(0.2)
        self.dense1 = keras.layers.Dense(units=256, activation='relu')
        self.dropout2 = keras.layers.Dropout(0.25)
        self.dense2 = keras.layers.Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dropout1(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x

model = employment()  # 实例化模型
#print(model.summary())  # 打印模型
# CPU版本
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.2)
#  -------------------------- 5、class类继承模型 -------------------------------


#  -------------------------- 6、保存模型，显示运行结果 -------------------------------
from keras.utils import plot_model

# 保存模型
model.save('model_MLP_text.h5')  # 生成模型文件 'my_model.h5'
# 模型可视化
plot_model(model, to_file='model_MLP_text.png', show_shapes=True)

from keras.models import load_model

# 加载模型
# model = load_model('model_MLP_text.h5')
print(x_train[0])
y_new = model.predict(x_train[0].reshape(1, 50))
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])

import matplotlib.pyplot as plt

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
#  -------------------------- 6、保存模型，显示运行结果 -------------------------------