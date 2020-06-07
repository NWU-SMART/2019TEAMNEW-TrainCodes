# ------------------------------作者信息-------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/26 21:30
# @Author: wangshengkang

# --------------------------------作者信息--------------------------------------
# --------------------------------代码布局---------------------------------------
# 1导入Keras，pandas，jieba，matplotlib，numpy的包
# 2招聘数据导入
# 3分词和提取关键词
# 4建立字典，并使用
# 5建立模型
# 6保存模型，画图
# -------------------------------代码布局-----------------------------------
# --------------------------------1导入相关包-----------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D, BatchNormalization
from keras.layers.embeddings import Embedding

# ---------------------------------1导入相关包-------------------------------------
# ----------------------------------2招聘数据导入--------------------------------------

job_detail_pd = pd.read_csv('../job_detail_dataset.csv', encoding='UTF-8')  # 读取文件
print(job_detail_pd.head(5))  # 打印前五行
label = list(job_detail_pd['PositionType'].unique())  # 将不重复的工作类型列出
print('label')
print(label)


def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引，将工作类型转化为数字标签
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)  # 加入label列
job_detail_pd = job_detail_pd.dropna()  # 删除空行
print('label_dataset')
print(job_detail_pd.head(5))


# ----------------------------------2招聘数据导入--------------------------------------
# ----------------------------------3分词和提取关键词----------------------------------
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))  # 中文分词


# 加入新的一列
job_detail_pd['Job Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
print('chinese_word_cut')
print(job_detail_pd.head(5))


# 提取关键词  topK=50，50个关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))


# 加入新的一列
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
print('key_word_extract')
print(job_detail_pd.head(5))

# ----------------------------------3分词和提取关键词----------------------------------
# ----------------------------------4建立字典----------------------------------
token = Tokenizer(num_words=2000)  # 建立2000个词的字典
# 按单词出现次数排序，排序前2000的单词列入词典中
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])
print('aaaaaaaaaaaaaaaaa\n',token)

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
print('bbbbbbbbbbbbbbbbbbbb',Job_Description_Seq)
# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
print('ccccccccccccccccccccccccc',Job_Description_Seq_Padding)
x_train = Job_Description_Seq_Padding  # 数字列表作为训练集
y_train = job_detail_pd['label'].tolist()  # 标签
print('x_train \n ',x_train.shape)
# ----------------------------------4建立字典----------------------------------
# ----------------------------------5建立模型----------------------------------
model = Sequential()
# Embedding层只能用在模型的第一层
model.add(Embedding(input_dim=2000,  # 字典大小
                    output_dim=32,  # 词向量的维度
                    input_length=50  # 每个数字列表的长度
                    )  # batch*50*32
          )  #
model.add(Conv1D(256,3,padding='same',activation='relu'))# 50*256 输出为256维，卷积核尺寸为3
model.add(MaxPool1D(3,3,padding='same'))# 17*256 池化窗尺寸为3，步长为3
model.add(Conv1D(32,3,padding='same',activation='relu')) # 17*32
model.add(Flatten())#544
model.add(Dropout(0.3))#544
model.add(BatchNormalization()) # 归一化#544
model.add(Dense(256,activation='relu'))#256
model.add(Dropout(0.2))#256
model.add(Dense(10,activation='softmax'))#10

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',
              # Computes the crossentropy loss between the labels and predictions.
              optimizer='adam',
              metrics=['accuracy']
              )

history = model.fit(
    x_train,
    y_train,
    batch_size=256,
    epochs=5,
    verbose=2,
    validation_split=0.2  # 训练集的20%用作验证集
)
# ----------------------------------5建立模型----------------------------------
# ----------------------------------6保存模型，画图----------------------------------
from keras.utils import plot_model

model.save('model_MLP_text.h5')  # 保存模型
plot_model(model, to_file='model_MLP_text.png', show_shapes=True)  # 模型可视化
print(x_train[0])#打印训练集第一行的数字序列
y_new = model.predict(x_train[0].reshape(1, 50))#第一行数据reshape为1行50列，然后预测
print(y_new) # 打印softmax得到的各个类别的概率
print(list(y_new[0]).index(max(y_new[0])))# 将softmax得到的最大概率索引取出作为预测结果
print(y_train[0])# 打印真实标签

import matplotlib.pyplot as plt

# 画准确率曲线
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

# 画损失函数曲线
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
# ----------------------------------6保存模型，画图----------------------------------
