# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月2日
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

#---------------------------------招聘数据导入--------------------------------

#数据路径
path = 'D:\\应用软件\\研究生学习\\job_detail_dataset.csv'     # 对比读取 npz数据方法   path = ''
#读取数据
data = pd.read_csv(path,encoding='utf-8')                    #  data = np.load(path)

print(data.head(5))     # 前五个
print(data.shape)       # 50000*2

#-------------------------------数据预处理----------------------

label = list(data['PositionType'].unique())
print(label)

# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row) #返回label列表对应值的索引
    return num_label

#给不同工作类型上数字标签  【0，项目管理】
data['label'] = data['PositionType'].apply(label_dataset)
#丢弃缺失部分
data = data.dropna()
print(data.head(5))

# 分词
def chinese_word(row):
    return" ".join(jieba.cut(row))
data['chinses_cut'] = data.Job_Description.apply(chinese_word)

#提取关键词
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

#建立字典
token = Tokenizer(num_words=1000)
#按照单词出现的顺序建立
token.fit_on_texts(data['keyword'])
#将文本转化为单词序列
description = token.texts_to_sequences(data['keyword'])
#将序列填充为最大的50个
job_description = sequence.pad_sequences(description,maxlen=50)

# 训练集
x_train = job_description
y_train = data['label'].tolist()   # ????????
#print(y_train.head(5))  不可这么写
#y_train = np.array(y_train)
#print(y_train)
x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)


#   ---------------------- 构建模型 ---------------------------
class Class_Model(nn.Module):  # 继承torch.nn.Module
    def __init__(self):  # 绑定两个属性
        super(Class_Model, self).__init__()
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


model = Class_Model()  # 实例化招聘模型
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


plt.plot(loss_list, 'r')
plt.show()
