'''
/------------------ 开发者信息 --------------------/
/** 开发者：于林生

开发日期：2020.5.22

版本号：Versoin 1.0

修改日期：2020.5.27

修改人：于林生

修改内容：将Pytorch的几种方法写入 /（将原来程序的错误改正(中间漏了一个层)）

/------------------ 开发者信息 --------------------*/
'''


# 导入torch需要的包
import torch
import torch.nn as nn
# 导入需要的数据处理包
# 导入需要的包
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


# 读取数据
# 数据路径
path = 'G:/python/code/多层感知器/keras_notebook代码/job_detail_dataset.csv'
# path = 'job_detail_dataset.csv'
# 读取数据
data = pd.read_csv(path,encoding='utf-8')
# 维度（50000 x 2 ）
# print(data)


# 数据处理
label = list(data['PositionType'].unique())#去重读取工作类型 10类
print(label)
print(label.index('项目管理'))#找到项目管理的索引（list的属性）
# 为工作描述设置标签的id
def label_dataset(row):
     num_label = label.index(row)  # 返回label列表对应值的索引
     return num_label

# 给不同的工作类型打上分类标签
data['label'] = data['PositionType'].apply(label_dataset)
# print(data)
data = data.dropna()
# print(data)#44831*3


# 提取描述中的中文分词并写入
# 采用的精确模式  他来到上海交通大学  ->   他/ 来到/ 上海交通大学
# (若参数cut_all=True)  ->  他/ 来到/ 上海/ 上海交通大学/ 交通/ 大学
def chinese_word(row):
    return " ".join(jieba.cut(row))
data['chinese_cut'] = data.Job_Description.apply(chinese_word)
data.head(5)


# 提取关键词
# 提取关键词
# analyse.extract_tags(texts,topK,withWeight,allowPOS)
'''
第一个参数：待提取关键词的文本
第二个参数：返回关键词的数量，重要性从高到低排序
第三个参数：是否同时返回每个关键词的权重
第四个参数：词性过滤，为空表示不过滤，若提供则仅返回符合词性要求的关键词
'''
def key_word_extract(texts):
  return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
data['keyword']=data.Job_Description.apply(key_word_extract)
data.head(5)


# 建立字典
token = Tokenizer(num_words=1000)
# 按照单词出现的顺序建立
token.fit_on_texts(data['keyword'])

description = token.texts_to_sequences(data['keyword'])
job_description = sequence.pad_sequences(description,maxlen=50)
# 选取训练集
x_train = job_description
y_train = data['label'].tolist()
# print(x_train)
# print(y_train)
# 将数据转换为longtensor型
y_train = torch.LongTensor(y_train)
x_train = torch.LongTensor(x_train)



# 建立模型
# /*------------------ 第一种利用类继承方式 --------------------*/
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=1000,embedding_dim = 32)
        self.dropout = torch.nn.Dropout(0.2)
        self.Flatten = torch.nn.Flatten()
        self.dense = torch.nn.Linear(in_features= 1600 ,out_features= 256)
        self.relu = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.25)
        self.softmax = torch.nn.Softmax()
        self.dense1 = torch.nn.Linear(in_features=256,out_features=10)
    #     前向传播
    def forward(self,data):
        data = self.embedding(data)
        data = self.dropout(data)
        data = self.Flatten(data)
        data = self.dense(data)
        data = self.relu(data)
        data = self.dropout2(data)
        data = self.dense1(data)
        y_predict = self.softmax(data)
        return y_predict
# /*------------------ 第二种利用sequential--------------------*/

# Net = torch.nn.Sequential(
#     torch.nn.Embedding(num_embeddings=1000,embedding_dim=32),
#     torch.nn.Dropout(0.2),
#     torch.nn.Flatten(),
#     torch.nn.Linear(in_features=1600,out_features=256),
#     torch.nn.ReLU(),
#     torch.nn.Dropout(0.25),
#     torch.nn.Linear(in_features=256,out_features=10),
#     torch.nn.Softmax()
# )
# /*------------------ 第三种利用sequential+ OrderedDict--------------------*/
# from collections import OrderedDict
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.model = torch.nn.Sequential(
#            OrderedDict([
#                ('embedding',torch.nn.Embedding(num_embeddings=1000,embedding_dim=32)),
#                ('dropout1',torch.nn.Dropout(0.2)),
#                ('flatten',torch.nn.Flatten()),
#                ('dense1',torch.nn.Linear(in_features=1600,out_features=256)),
#                ('relu',torch.nn.ReLU()),
#                ('dropout2',torch.nn.Dropout(0.25)),
#                ('dense2',torch.nn.Linear(in_features=1600,out_features=256)),
#                ('softmax',torch.nn.Softmax())
#
#             ]))
#     def forward(self,x):
#         result = self.model(x)
#         return x
# /*------------------ 第四种利用类继承add_module的方式--------------------*/
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.input = torch.nn.Sequential()
#         self.input.add_module('embedding',torch.nn.Embedding(num_embeddings=1000,embedding_dim=32))
#         self.input.add_module('dropout1',torch.nn.Dropout(0.2))
#         self.input.add_module('flatten',torch.nn.Flatten())
#         self.input.add_module('dense',torch.nn.Linear(in_features=1600,out_features=256))
#         self.input.add_module('dropout2',torch.nn.Dropout(0.25))
#         self.input.add_module('relu',torch.nn.ReLU())
#         self.input.add_module('softmax',torch.nn.Softmax())
#     def forward(self,x):
#         result = input(x)
#         return result



#     调用模型
model = Net()
# 选择交叉熵损失函数
loss_f = nn.CrossEntropyLoss()
# 调用SGD优化函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 定义训练批次
epoch = 10
for i in range(epoch):
    # 调用模型预测（有点warning可以正常运行，torch版本太高容易报softmax的warning）
    y_pred = model(x_train)
    # 求损失
    loss = loss_f(y_pred,y_train)
    # 梯度置零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 权值更新
    optimizer.step()
#     画出预测值和真实值之间的区别
    print(i, loss.item())



