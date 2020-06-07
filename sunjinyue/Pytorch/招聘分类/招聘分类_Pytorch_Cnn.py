# ----------------开发者信息--------------------------------#
# 开发者：孙进越
# 开发日期：2020年6月4日
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

# 因为是Pytorch方法所以需要将数据转化为Tensor格式

x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)
#-------------------------数据预处理---------------------

#---------------------模型-------------------------------
class CnnModel(torch.nn.Module):
    def __init__(self):
        super(CnnModel,self).__init__()
        self.dense = torch.nn.Sequential(torch.nn.Embedding(num_embeddings=1000, embedding_dim=32),
                                         torch.nn.Conv1d(in_channels=50,out_channels=256,kernel_size=5,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool1d(kernel_size=3,padding=1),
                                         torch.nn.Conv1d(in_channels=256, out_channels=50, kernel_size=5, padding=1,
                                                         stride=1),
                                         torch.nn.Flatten(),
                                         torch.nn.BatchNorm1d(350),  
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(350, 256),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(256, 10),
                                         torch.nn.Softmax()
                                         )

    def forward(self, x):
            x = self.dense(x)
            return x
model = CnnModel() # 实例化
print(model) #打印

optimizer = torch.optim.SGD(model.parameters(),lr=0.001) #优化器
loss_func = torch.nn.CrossEntropyLoss() #损失函数 均方误差

#  ---------------------------训练------------------------------
loss_list = [] #需要将loss存入列表中，所以建立一个列表
for t in  range(5):
    train_Prediction = model(x_train)
    loss = loss_func(train_Prediction,y_train) #带入损失函数计算损失
    loss_list.append(loss) #append加入list

    optimizer.zero_grad() #Pytorch 需要在每一个batch手动清零梯度
    loss.backward() #反向传播，算参数
    optimizer.step() #参数更新
    print(loss)


plt.plot(loss_list,'r')
plt.show()
