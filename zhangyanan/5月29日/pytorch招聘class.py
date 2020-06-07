# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年5月29日
# 修改日期：
# 修改人：
# 修改内容：


#  -------------------------- 导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#   ---------------------- 数据的载入和处理 ----------------------------
job_detail_pd = pd.read_csv('D:\\Study\\project\\招聘信息文本分类\\job_detail_dataset.csv', encoding='UTF-8')
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

x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)


#   ---------------------- 构建模型 ---------------------------
class JobModel(nn.Module):  # 继承torch.nn.Module
    def __init__(self):  # 绑定两个属性
        super(JobModel, self).__init__()
        self.embedding = torch.nn.Embedding(2000, 32)
        self.dropout = torch.nn.Dropout(0.2)
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(1600, 256)  # 经过flatten 50*32
        self.relu = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(256, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x


model = JobModel()  # 实例化招聘模型
print(model)  # 打印模型结构

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器为SGD，学习率是1e-3
loss_func = torch.nn.CrossEntropyLoss()   # 定义损失函数为均方误差


#   ---------------------- 训练模型 ---------------------------
loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    train_prediction = model(x_train)
    loss = loss_func(train_prediction, y_train)  # 计算损失
    loss_list.append(loss) # 使用append()方法把每一次的loss添加到loss_list中

    optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
    loss.backward()  # 反向传播，计算参数
    optimizer.step()  # 更新参数
    print(loss)


plt.plot(loss_list, 'r-')
plt.show()
