'''
/------------------ 开发者信息 --------------------/
/** 开发者：于林生

开发日期：2020.5.29

版本号：Versoin 1.0

修改日期：

修改人：

修改内容：

/------------------ 开发者信息 --------------------*/
'''

# /------------------ 代码布局 --------------------*/
'''
代码布局
1.导入需要的包
2.读取数据
3.数据预处理
4.建立模型
5.模型训练
'''
# /------------------ 代码布局 --------------------*/

# /------------------ 导入需要的包 --------------------*/
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

# /------------------ 导入需要的包 --------------------*/


# /------------------ 读取数据 --------------------*/

# 读取数据
# 数据路径
path = 'G:/python/code/多层感知器/keras_notebook代码/job_detail_dataset.csv'
# path = 'job_detail_dataset.csv'
# 读取数据
data = pd.read_csv(path,encoding='utf-8')
# 维度（50000 x 2 ）
# print(data)

# /------------------ 读取数据 --------------------*/

# /------------------ 数据处理 --------------------*/

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
# /------------------ 数据处理 --------------------*/

# /------------------ 建立模型 --------------------*/
# 引入需要的层
from torch.nn import Embedding,Linear,MaxPool1d,Conv1d,BatchNorm1d,Dropout,ReLU,Softmax
class cnn(torch.nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.embedding = Embedding(num_embeddings=1000,embedding_dim = 32,)#嵌入层嵌入大小，嵌入维度，输入单词的长度50
        self.conv1 = Conv1d(in_channels=50,out_channels=256,kernel_size=3,padding=1)#输入50，输入256,
        self.maxpool = MaxPool1d(kernel_size=3,padding=1)
        self.conv2 = Conv1d(256,50,kernel_size=3,padding=1)
        self.flatten = torch.nn.Flatten()
        self.batchnormal = BatchNorm1d(550)#不是很清楚这个维度是怎么确定的（报错显示的维度是转换为550）
        self.dense1 = Linear(550,256)
        self.dense2 = Linear(256,10)
        self.dropout = Dropout(0.3)
        self.relu = ReLU()
        self.softmax = Softmax()
    def forward(self,x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.batchnormal(x)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        result = self.softmax(x)
        return x
model = cnn()
# 优化器使用adam
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
# 损失使用交叉熵损失
loss_fn = torch.nn.CrossEntropyLoss()
# 模型训练
epoch = 5
for i in range(epoch):
    # 预测结果
    y_pred = model(x_train)
    # 计算损失
    loss = loss_fn(y_pred,y_train)
    # 梯度归零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 梯度更新
    optimizer.step()
    print(i,loss.item())
     # if max(loss.item):
    #     torch.save(model.state_dict(),'cnn.pkl')

