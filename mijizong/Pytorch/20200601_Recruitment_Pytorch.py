# ----------------------开发者信息-----------------------------------------
 # -*- coding: utf-8 -*-
 # @Time: 2020/6/1
 # @Author: MiJizong
 # @Version: 1.0
 # @Version: 1.0
 # @FileName: 1.0.py
 # @Software: PyCharm
 # ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 jieba, Keras, torch, matplotlib, numpy, sklearn 和 panda的包
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
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 数据目录:  D:\Office_software\PyCharm\keras_datasets\job_detail_dataset.csv
job_detail_pd = pd.read_csv('D:\\Office_software\\PyCharm\\keras_datasets\\job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))                            # 打印数据的前五行
label = list(job_detail_pd['PositionType'].unique())    # 提取所有岗位标签（非重复）
print(label)                                            # 打印标签

# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)   # 将所有岗位与数字标签相对应
job_detail_pd = job_detail_pd.dropna()                                        # 删除空行
job_detail_pd.head(5)                                                         # 展示前五行数据
#  -------------------------- 2、招聘数据数据导入 -------------------------------


#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):               # 使用jieba分词器进行分词
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut) # 执行分词
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=())) # 使用默认的TF-IDF模型对文档进行分析
                                    # 参数withWeight设置为True时可以显示词的权重，topK设置显示的词的个数

job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
#  -------------------------- 3、分词和提取关键词 -------------------------------


#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现频率排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”  将中文描述转化为数字化标签
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()
y_train = np.array(y_train)  #list格式转换成array格式

x = Variable(torch.from_numpy(x_train)).long()      # x_train变为variable数据类型
y = Variable(torch.from_numpy(y_train)).long()      # y_train变为variable数据类型
#  -------------------------- 4、建立字典，并使用 -------------------------------


#  ----------------------------- 5.1、Sequential模型构造 ---------------------------
class Recruitment1(nn.Module):
    def __init__(self):
        super(Recruitment1,self).__init__()
        self.dense = nn.Sequential(
                    nn.Embedding(num_embeddings=2000, embedding_dim=32, max_norm=50),
                    nn.Dropout(0.2),
                    nn.Flatten(),
                    nn.Linear(1600, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 10),
                    nn.Softmax())

    def forward(self, x):
        x = self.dense(x)
        return x
#  ---------------------------- 5.1、Sequential模型构造 -----------------------------

#  ----------------------------- 5.2、another_method模型构造 -----------------------------
class Recruitment2(nn.Module):
    def __init__(self):
        super(Recruitment2,self).__init__()
        self.conv = nn.Sequential()
        self.conv.add_module('embedding', torch.nn.Embedding(2000,32))
        self.conv.add_module('conv1',nn.Conv1d(256, 3, kernel_size=3))
        self.conv.add_module('relu1',nn.ReLU())
        self.conv.add_module('maxpool',nn.MaxPool1d(3,3,padding='same'))
        self.conv.add_module('conv2',nn.Conv1d(32, 3, kernel_size=3))
        self.conv.add_module('relu1', nn.ReLU())
        self.conv.add_module('flatten',nn.Flatten())
        self.conv.add_module('dropout1',nn.Dropout(0.3))
        self.conv.add_module('dense1',nn.Linear(32,256))
        self.conv.add_module('dropout2',nn.Dropout(0.2))
        self.conv.add_module('dense2',nn.Linear(256,10))

    def forward(self, x):
        x = self.conv(x)
        return x
#  ----------------------------- 5.2、another_method模型构造 -----------------------------

#  ----------------------------- 5.3、class继承模型构造 -----------------------------
class Recruitment3(torch.nn.Module):
    def __init__(self):
        super(Recruitment3, self).__init__()
        self.embedding = torch.nn.Embedding(2000,32)
        self.dropout1 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(1600, 256)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(256,10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x
#  ----------------------------- 5.3、class继承模型构造 -----------------------------


#  -------------------------- 6、模型训练 -----------------------------------------
model = Recruitment1()
print(model)

loss_func = nn.CrossEntropyLoss()                        # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)  # SGD优化器

loss_list = []                                           # 存放loss
loss_item = []                                          # 记下每个epoch
EPOCH = 5
for epoch in range(EPOCH):
    prediction = model(x)
    loss = loss_func(prediction, y)                     # 计算损失
    loss_item.append(epoch)
    loss_list.append(loss)                              # 将每一次的loss添加到loss_list中
    optimizer.zero_grad()                               # 梯度归零
    loss.backward()                                     # 反向传播
    optimizer.step()                                    # 梯度更新
    print(epoch)
    print(loss)
#  -------------------------- 6、模型训练 -----------------------------------------


#  -------------------------- 7、模型可视化 ---------------------------------------
plt.plot(loss_item, loss_list, label="Train loss")
plt.plot()
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')  # 给图加上图例
plt.savefig('test.png')                      # 保存
plt.show()
#  ---------------------------- 7、模型可视化 ---------------------------------------