# ----------------开发者信息------------------------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月3日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息------------------------------------------------------
# ----------------------   代码布局： --------------------------------------------
# 1、导入程序所需pytorch的包
# 2、招聘数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： -------------------------------------------
# #  -------------------------- 1、导入需要包 -----------------------------------
import torch
import torch.nn as nn
# 导入需要的数据处理包
import pandas as pd #数据处理包
import jieba #中文词库包
import jieba.analyse as analyse
import numpy as np
import matplotlib.pyplot as plt #画图包
from sklearn.model_selection import train_test_split #数据划分包
from sklearn.preprocessing import LabelEncoder  #数据有类别编码
import torchtext
from keras.preprocessing.text import Tokenizer  #将文本处理成索引类型的数据
from keras.preprocessing import sequence
from torch.autograd import Variable
#  -------------------------- 1、导入需要包 --------------------------------------
#  -------------------------- 2、招聘数据数据导入 --------------------------------
path = 'D:/northwest/小组视频/3招聘文本分类CNN/job_detail_dataset.csv'
data = pd.read_csv('path',encoding='UTF-8')#编码为utf-8
label = list(job_detail_pd['Positiontype']).unique #标签
print(label)
def label_dataset(row):#为工作描述设置标签
    num_label = label.index(row)
    return num_label
data['label'] = data['PositionType'].apply(label_dataset)
data = data.dropna()#删除空行
#  -------------------------- 2、招聘数据数据导入 --------------------------------
#  -------------------------- 3、分词和提取关键词 --------------------------------
#中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))
data['chinese_cut'] = data.Job_Description.apply(chinese_word)
data.head(5)
#提取关键字
def key_word_extract(row):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
data['key_extract'] = data.Job_Description.apply(key_word_extract)
data.head(5)
#  -------------------------- 3、分词和提取关键词 --------------------------------
#  -------------------------- 4、建立字典，并使用 --------------------------------
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['key_extract'])#按单词出现次数排序，排序前2000的单词会列入词典中
Job_Description_Seq = token.text_to_sequences(job_detail_pd['key_extract'])#将文字转换为数字列表
Job_Description_Seq_Padding = sequence.pad.sequences(Job_Description_Seq,maxlen=50)#截长补短，让所有数字列表长度均为50
x_train = Job_Description_Seq
y_train = data['label'].tolist#数组转列表
#  -------------------------- 4、建立字典，并使用 --------------------------------
#  -------------------------- 5、训练模型 ---------------------------------------
#/--------------------------method1-class-----------------------------------------/#
class Model1(nn.Model):
    def __init__(self):
        super(Model1,self).__init()
        self.embedding = torch.nn.Embedding(output_dim=32,intput_dim=2000,input_length=50)
        self.conv1 = torch.nn.Conv1D(256,3,padding='same',activation='relu')
        self.maxpooling = torch.nn.Maxpooling(3,3,padding='same')#在图片周围尽量均匀的填充0,用来满足剩余图像部分不够卷积核扫描的情况
        self.conv2 = torch.nn.Conv1D(3,3,padding='same',activation='relu')
        self.flatten = torch.nn.Flatten()
        self.dropout1 = torch.nn.Dropout(0.3)
        self.batchnormal = torch.nn.BatchNorm1d(550)
        self.dense1 = torch.nn.Dense(256, activation='relu')
        self.dropout2 = torch.nn.Droput(0.2)
        self.dense2 = torch.nn.Dense(10, activation='softmax')
    def forward(self,x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.maxpooling (x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.batchnormal(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return(x)
#/--------------------------method1-class-----------------------------------------/#
#/--------------------------method2-Sequential------------------------------------/#
class Model2(nn.Model):
    def __init__(self):
        super(Model2,self).__init()
        self.layer = torch.nn.Sequential(
            torch.nn.Embedding(output_dim=32,intput_dim=2000,input_length=50),
            torch.nn.Conv1D(256, 3, padding='same', activation='relu'),
            torch.nn.Maxpooling(3, 3, padding='same') ,
            torch.nn.Conv1D(3, 3, padding='same', activation='relu'),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.3),
            torch.nn.BatchNorm1d(550),
            torch.nn.Dense(256, activation='relu'),
            torch.nn.Droput(0.2),
            torch.nn.Dense(10, activation='softmax')
        )
    def call(self,x):
        x = self.layer(x)
        return x
#/--------------------------method2-Sequential------------------------------------/#
model = Model1()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)# 优化器使用adam
loss_fn = torch.nn.CrossEntropyLoss()
epoch = 5
for i in range(epoch):
    y_p = model(x_train)
    loss = loss_fn(y_p,y_train)
    optimizer.zero_grad()#梯度归零
    loss.backward()
    optimizer.step()
    print(i, loss.item())
#  -------------------------- 5、训练模型 -----------------------------------------









