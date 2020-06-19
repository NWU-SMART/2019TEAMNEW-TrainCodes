# ----------------开发者信息----------------------------
# 开发者：张涵毓
# 开发日期：2020年5月29日
# 内容：招聘信息文本分类-class pytorch
# 修改者：
# ----------------开发者信息----------------------------
# ----------------------代码布局-------------------------------------
# 1.引入keras，matplotlib，numpy，sklearn，pandas,jieba包
# 2.导入招聘信息数据
# 3.分词和提取关键词
# 4.建立和使用字典
# 5.训练模型
# 6.保存模型，分类结果
# ---------------------------------------------------------------------

#-------------------------1、导入相关包-------------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers  # 正则化
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization


#-------------------------1、导入相关包-------------------------------------
#--------------------------2、导入招聘信息数据-------------------------------
path='D:\\研究生\\代码\\Keras代码\\1.Multi-Layer perceptron(MLP 多层感知器)\\job_detail_dataset.csv'
job_detail_pd = pd.read_csv(path, encoding='UTF-8')
print(job_detail_pd.heas(5)) #显示前5个数据
label = list(job_detail_pd['PositionType'].unique())  # 标签
print(label)

# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label

job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)
#--------------------------2、导入招聘信息数据-------------------------------
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
# keras.Tokenizer
# 参数num_words：需要保留的最大词数，基于词频。只有最常出现的 num_words 词会被保留


token.fit_on_texts(job_detail_pd['Job_Description_key_word'])
#fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。
# 使用token字典将“文字”转化为“数字列表”

Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
#texts_to_sequences(texts)
#将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)

x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist() #tolist生成列表，但和list不同
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  -------------------------- 5、训练模型 -------------------------------
batch_size=256
epochs=5
#----------------------------module1---------------------------------
#import torch.nn as nn
class MLPic1(nn.module):
    def __init__(self):
        super(MLPic1, self).__init__()
        self.embedding =nn.Embedding(output_dim=32, input_dim=2000)
        self.dropout1 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Dense(units=256, activation='relu')
        self.dropout2 = nn.Dropout(0.25)
        self.dense2 = nn.Dense(units=10, activation='softmax')
#前向传递，输出结果
    def forward(self, x):
            x = self.embedding(x)
            x = self.dropout1(x)
            x = self.flatten(x)
            x = self.dense1(x)
            x = self.dropout2(x)
            y_result = self.dense2(x)
            return y_result

model = MLPic1( )
loss_fn=nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

loss_list = []
for i in range(epochs):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)  # 计算损失
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新梯度
    print(loss)
plt.plot(loss_list, 'r-')
plt.show()


'''
# 绘制训练 & 验证的准确率值
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
'''
