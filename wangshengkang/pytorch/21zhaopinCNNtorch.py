# ----------------------------作者信息--------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/27 11:15
# @Author: wangshengkang

# ----------------------------作者信息-----------------------------------
# --------------------------------代码布局---------------------------------------
# 1导入Keras，pandas，jieba，matplotlib，numpy的包
# 2招聘数据导入
# 3分词和提取关键词
# 4建立字典，并使用
# 5建立模型
# 6保存模型，画图
# -------------------------------代码布局-----------------------------------
# --------------------------------1导入相关包-----------------------------------
import torch
import torch.nn as nn
import pandas as pd
import jieba
import jieba.analyse as analyse
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import matplotlib.pyplot as plt

# --------------------------------1导入相关包-----------------------------------
# ----------------------------------2招聘数据导入--------------------------------------

job_detail_pd = pd.read_csv('job_detail_dataset.csv', encoding='UTF-8')  # 读取文件
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


# 提取关键词
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

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding  # 数字列表作为训练集
y_train = job_detail_pd['label'].tolist()  # 标签

x_train = torch.LongTensor(x_train)  # 转化为tensor形式
y_train = torch.LongTensor(y_train)


# TypeError: embedding(): argument 'indices' (position 2) must be Tensor, not numpy.ndarray

# ----------------------------------4建立字典----------------------------------
# ----------------------------------5建立模型----------------------------------

class zhaopin(nn.Module):
    def __init__(self):
        super(zhaopin, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=2000, embedding_dim=32)  # 50*32
        self.model = nn.Sequential(
            nn.Conv1d(32, 256, 3, padding=2),  # 52*256
            nn.ReLU(),
            nn.MaxPool1d(3, 3),  # 17*256
            nn.Conv1d(256, 32, 3, padding=1),  # 17*32
            nn.ReLU(),
            nn.Flatten(),  # 544
            nn.Dropout(0.3),
            nn.BatchNorm1d(544),  # 544
            nn.Linear(544, 256),  # 256
            nn.Dropout(0.2),
            nn.Linear(256, 10),  # 10
            nn.Softmax()
        )

    def forward(self, x):
        out = self.embedding(x)
        # RuntimeError: Given groups=1, weight of size [256, 32, 3], expected input[44831, 50, 32] to have 32 channels, but got 50 channels instead
        out = out.permute(0, 2, 1)
        out = self.model(out)

        return out


model = zhaopin()
print(model)
epochs = 5
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
iteration = []  # list存放epoch数
loss_total = []  # list存放损失
for epoch in range(epochs):
    pre = model(x_train)
    train_loss = loss(pre, y_train)
    iteration.append(epoch)  # 将epoch放到list中
    loss_total.append(train_loss)  # 将loss放到list中
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    print('epoch %3d, loss %10.7f' % (epoch, train_loss))

print('loss_total\n', loss_total)
plt.plot(iteration, loss_total, label="loss")  # iteration和loss对应画出
plt.title('torch loss')  # 题目
plt.xlabel('Epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train'], loc='upper left')  # 图线示例
plt.show()  # 画图

torch.save(model.state_dict(), "zhaopincnntorch.pth")  # 保存模型参数
# model.load_state_dict(torch.load('housetorch.pth'))  # 加载模型
model.eval()  # 评估模式
print(x_train[0])  # 打印训练集第一行的数字序列
y_new = model(x_train[0].reshape(1, 50))  # 第一行数据reshape为1行50列，然后预测
print(y_new)  # 打印softmax得到的各个类别的概率
print(list(y_new[0]).index(max(y_new[0])))  # 将softmax得到的最大概率索引取出作为预测结果
print(y_train[0])  # 打印真实标签
