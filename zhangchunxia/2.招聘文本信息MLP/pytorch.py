# ----------------开发者信息------------------------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月1日
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
# ----------------------   代码布局： --------------------------------------------

#  -------------------------- 1、导入需要包 --------------------------------------
from collections import OrderedDict

import torch
import torch.nn as nn
import pandas as pd #数据处理包
import jieba #中文词库包
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer #将文本处理成索引类型的数据
from keras.preprocessing import sequence
import numpy as np
from sklearn.model_selection import train_test_split #数据划分包
from sklearn.preprocessing import LabelEncoder #数据有类别编码
import torchtext
from torch.autograd import Variable #给定一个输入x,autograd会给其自动求y关于x的微分，得出其输出y
#  -------------------------- 1、导入需要包 --------------------------------------

#  -------------------------- 2、招聘数据数据导入 --------------------------------
job_detail_pd = pd.read_csv('D:/northwest\小组视频/2招聘文本信息MLP/job_detail_dataset.csv')#数据目录
print(job_detail_pd.head(5))#读前五行
label = list(job_detail_pd['Position'].unique())#职位标签
print(label)#打印标签
#为工作描述设置标签的ID
def label_dataset(row):
    num_label = label.index(row) #返回label列表对应值的索引
    return num_label
job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset())#不同的工作类型打标签
job_detail_pd = job_detail_pd.dropna()#删除多余的空行，删除丢失的部分
print(job_detail_pd.head(5))#打印前五行
#  -------------------------- 2、招聘数据数据导入 --------------------------------

#  -------------------------- 3、分词和提取关键词 --------------------------------
def chinese_word_cut(row):
    return"".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Descriptin.apply(chinese_word_cut)#apply函数的返回值就是chinese_word_cut函数的返回值
print(job_detail_pd.head(5))
#提取关键字
def key_word_extract(texts):
    return "".join(analyse.extract_tags(texts,topk=50,),withWeight=False,allowPOS=())#提取50个
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Descriptin.apply(key_word_extract)
print(job_detail_pd.head(5))
#  -------------------------- 3、分词和提取关键词 ---------------------------------

#  -------------------------- 4、建立字典，并使用字典 -----------------------------
#建立字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq,maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist() #数组转为列表
#要使用autograd需要将数据转换为Variable数据类型
x_train = Variable(torch.from_numpy(x_train)).long()
y_train = Variable(torch.from_numpy(y_train)).long()
#  -------------------------- 4、建立字典，并使用字典 -----------------------------

#  -------------------------- 5、模型建立-----------------------------------------
#/--------------------------method1-class-----------------------------------------/#
class Model1(nn.Model):
    def __init__(self):
        super(Model1,self).__init()
        self.embedding = torch.nn.Embedding(num_embedding=2000,embedding_dim=32)
        self.dropout = torch.nn.Drpout(0.2)
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(in_features=1600,out_features=256)
        self.relu = torch.nn.Relu()
        self.dropout = torch.nn.Dropout(0.25)
        self.dense2 = torch.nn.Linear(in_features=256,out_features=10)
        self.softmax = torch.nn.Softmax()
    def forward(self,x):
        x = self.embedding(x)
        x = self.dropout(0.2)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(0.25)
        x = self.dense2(x)
        x = self.softmax(x)
        return x
#/--------------------------method1-class-----------------------------------------/#
#/--------------------------method2-Sequential------------------------------------/#
class Model2(nn.Model):
    def __init__(self):
        super(Model2,self).__init()
        self.layer = torch.nn.Sequential(
            torch.nn.Embedding(2000,32),
            torch.nn.Dropout(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(1600,256),
            torch.nn.relu(),
            torch.nn.dropout(0.25),
            torch.nn.Linear(256,10),
            torch.nn.softmax()
        )
    def forward(self,x):
        x = self.layer(x)
        return x
# /--------------------------method2-Sequential------------------------------------/#
#/-------------------------------method3-------------------------------------------/#
#class Model3(nn.Model):
    #def __init__(self):
       # super(Model3,self).__init()
       # self.model = torch.nn.Sequential(
       #    OrderedDict([
       #         'embedding',torch.nn.Embedding(num_embedding=2000,embedding_dim=32),
       #         'dropout',torch.nn.Dropout(0.2),
       #         'flatten',torch.nn.Flatten(),
       #         'dense1',torch.nn.Linear(in_features=1600,out_features=256),
       #         'relu',torch.nn.Relu(),
       #         'dropout',torch.nn.Dropout(0.25),
       #         'dense2',torch.nn.Linear(in_features=256,out_features=10)
       #         'softmax',torch.nn.Softmax()
       #     ])
       #  )
  #   def forward(self,x):
  #      x = self.model(x)
  #      return x
#/-------------------------------method3-------------------------------------------/#
#/--------------------------method4-add module-------------------------------------/#
class Model4(nn.Model):
    def __init__(self):
        super(Model4,self).__init()
        self.dense = torch.nn.Sequential(
            self.dense.add.module('embedding',torch.nn.Embedding(num_embedding=2000,embedding_dim=32)),
            self.dense.add.module('dropout',torch.nn.Dropout(0.2)),
            self.dense.add.module('flatten',torch.nn.Flatten),
            self.dense.add.module('dense1',torch.nn.Linear(in_features=1600,out_features=256)),
            self.dense.add.module('relu',torch.nn.Relu()),
            self.dense.add.module('dropout',torch.nn.Dropout(0.25)),
            self.dense.add.module('dense2',torch.nn.Linear(in_features=256,out_features=10)),
            self.dense.add.module('softmax',torch.nn.Softmax())
        )
    def forward(self,x):
        x = self.dense(x)
        return x
#/--------------------------method4-add module--------------------------------------/#
#  -------------------------- 5、模型建立-----------------------------------------

#  -------------------------- 6、模型训练及保存 ------------------------------------
model = Model1()
print(model)
optimizer = torch.optim.Adam(model.parameters,lr=1e-4)
loss_fn = torch.CrossEntropyLoss()
Epoch = 8
for i in range(Epoch):
    y_p = x_train()
    loss = loss_fn(y_p,y_train)
    optimizer.zero.grad()
    loss.backward()
    optimizer.step()#权值更新
#保存模型
print('模型保存')
torch.save(Model1,'mlp,zp')
#模型加载
model = torch.load('mlp,zp')
#  -------------------------- 6、模型训练及保存 ------------------------------------

