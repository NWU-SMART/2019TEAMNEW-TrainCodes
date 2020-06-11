# ----------------------开发者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/6/3
# @Author: MiJizong
# @Content:  招聘信息文本分类CNN——Pytorch三种方法实现
# @Version: 1.0
# @FileName: 1.0.py
# @Software: PyCharm
# ----------------------开发者信息-----------------------------------------

# ----------------------   代码布局： --------------------------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： ---------------------------------------

#  -------------------------- 1、导入需要包 ----------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#  -------------------------- 1、导入需要包 ------------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 文件放置在目录  D:\Office_software\PyCharm\keras_datasets\job_detail_dataset.csv
from numpy import shape

job_detail_pd = pd.read_csv('D:\\Office_software\\PyCharm\\keras_datasets\\job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))
label = list(job_detail_pd['PositionType'].unique())  # 提取非重复标签
print(label)


# 为工作描述设置每一行标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)  # 将所有岗位与数字标签相对应
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)  # 展示前五行数据


#  -------------------------- 2、招聘数据数据导入 -------------------------------

#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):  # 使用jieba分词器进行分词
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)  # 执行分词
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
    # 使用默认的TF-IDF模型对文档进行分析  参数withWeight设置为True时可以显示词的权重，topK设置显示的词的个数


job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)  # 执行提取
#  -------------------------- 3、分词和提取关键词 -------------------------------

#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”  将中文描述转化为数字化标签
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()

# 转换为tensor形式
x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  ----------------------------- 5.1、Sequential训练模型 -----------------------
class Recruitment1(nn.Module):
    def __init__(self):
        super(Recruitment1,self).__init__()
        #self.linear = nn.Embedding(num_embeddings=2000, embedding_dim=32, max_norm=50)  # 50 * 32
        self.dense = nn.Sequential(
                    nn.Embedding(num_embeddings=2000, embedding_dim=32, max_norm=50),  # 50 * 32
                    nn.Conv1d(in_channels=50,out_channels=256,kernel_size=3,padding=1),  # 52 * 256
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=3,stride=3,padding=1),  # 17 * 256
                    nn.Conv1d(in_channels=256,out_channels=50,kernel_size=3,padding=1), # 17 * 32
                    nn.ReLU(),
                    nn.Flatten(),  # 544
                    nn.Dropout(0.3),
                    nn.BatchNorm1d(num_features=550),
                    nn.Linear(in_features=550,out_features=256),  # 256
                    nn.Dropout(0.2),
                    nn.Linear(256,10),  # 10
                    nn.Softmax())

    def forward(self, x):
        #x = self.linear(x)
        #x = x.permute(0, 2, 1) # 需要对输入调整一下参数，使得通道在第二个
        x = self.dense(x)

        return x

#  ----------------------------- 5.1、Sequential训练模型 -----------------------

#  ----------------------------- 5.2、class继承模型构造 -----------------------------
class Recruitment2(torch.nn.Module):
    def __init__(self):
        super(Recruitment2, self).__init__()
        self.embedding = nn.Embedding(2000,32)
        self.conv1 = nn.Conv1d(50,256,3,padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3,3,1)
        self.conv2 = nn.Conv1d(256,50,3,padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)
        self.batchnorm = nn.BatchNorm1d(550)
        self.linear1 = nn.Linear(550,256)
        self.dropout2 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(256,10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.batchnorm(x)
        x = self.linear1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
#  ----------------------------- 5.2、class继承模型构造 -----------------------------

#  -------------------------- 6、模型训练 ------------------------------------------
model = Recruitment2()
print(model)


loss_func = nn.CrossEntropyLoss()                        # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)  # SGD优化器

loss_list = []                                           # 存放loss
loss_item = []                                          # 记下每个epoch
EPOCH = 5
for epoch in range(EPOCH):
    prediction = model(x_train)
    loss = loss_func(prediction, y_train)               # 计算损失
    loss_item.append(epoch)                             # 记录每次的epoch
    loss_list.append(loss)                              # 将每一次的loss添加到loss_list中
    optimizer.zero_grad()                               # 梯度归零
    loss.backward()                                     # 反向传播
    optimizer.step()                                    # 梯度更新
    print(epoch)
    print(loss)
#  -------------------------- 6、模型训练 -----------------------------------------


#  -------------------------- 7、模型可视化 ---------------------------------------
plt.plot(loss_item, loss_list, label="Train loss")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')  # 给图加上图例
plt.savefig('test.png')                      # 保存
plt.show()

torch.save(model.state_dict(), "RecruitmentCNNTorch.pth")  # 保存模型参数
# model.load_state_dict(torch.load('housetorch.pth'))   # 加载模型
model.eval()                               # 评估模式
print("打印训练集第一行的数字序列")
print(x_train[0])                           # 打印训练集第一行的数字序列
y_new = model(x_train[0].reshape(1, 50))   # 第一行数据reshape为1行50列，然后预测
print("打印softmax得到的各个类别的概率y_new")
print(y_new)                                # 打印softmax得到的各个类别的概率
print("打印预测结果")
print(list(y_new[0]).index(max(y_new[0])))  # 将softmax得到的最大概率索引取出作为预测结果
print("打印真实标签")
print(y_train[0])  # 打印真实标签
#  ---------------------------- 7、模型可视化 ---------------------------------------
