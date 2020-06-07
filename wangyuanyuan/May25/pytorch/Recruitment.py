#-------------------------------开发者信息----------------------------------
#开发人：王园园
#日期：2020.5.25
#开发软件：pycharm
#项目：招聘信息文本分类（pytorch）:只对模型进行了pytorch改写
#注：如果有看这个代码的同学，希望能帮我指正错误

#--------------------------------------------导入包-----------------------------------------------
from collections import OrderedDict
from msilib import sequence
from sre_parse import Tokenizer
from catalyst.utils import torch
import torch.nn.functional as F
from keras import Input
from numpy import shape
import pandas as pd

#--------------------------------------------招聘数据导入-----------------------------------------
job_details = pd.read_csv('',enconding='UTF-8')            #数据
label = list(job_details['PositionType'].unique())         #标签

#为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)                              #返回label列表对应值的索引
    return num_label

job_details['label']= job_details['PositionType'].apply(label_dataset)
job_details = job_details.dropna()  #删除空行

#------------------------------------------分词和提取关键词------------------------------------------
#中文分词
def chinese_word_cut(row):
    return ' '.join('jieba'.cut(row))

job_details['Job_Description_jieba_cut'] = job_details.Job_Description.apply(chinese_word_cut)

#提取关键词
def key_word_extract(texts):
    return ' '.join('analyse'.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))

job_details['Job_Description-_key_word'] = job_details.Job_Description.apply(key_word_extract)

#--------------------------------------建立字典，并使用----------------------------------------
#建立2000个词的字典
token = Tokenizer(num_words = 2000)
#按单词出现次数排序，排序前2000个单词会列入词典中
token.fit_on_texts(job_details['Job_Description-_key_word'])
#使用token字典将‘文字’转化为‘数字列表’
Job_Description_Seq = token.texts_to_sequences(job_details['Job_Description-_key_word'])
#截长补短让所有‘数字列表’长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding      #训练数据
y_train = job_details['lable'].tolist()    #训练标签

#-----------------------------------------构建模型-----------------------------------------
#-----------------------------------------Method1-----------------------------------------
input = Input(shape())
class model1(torch.nn.Module):
    def __init__(self):
        super(model1, self).__init__()
        self.embedding = torch.nn.Embedding(output_dim=32,
                                            input_dim=2000,
                                            input_length=50)
        self.conv1 = torch.nn.Conv1d(256, 3, padding='same', activation='relu')
        self.conv2 = torch.nn.Conv1d(32, 3, padding='same', activation='relu')
        self.dense1 = torch.nn.Linear(32, 256)
        self.dense2 = torch.nn.Linear(256,10)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.batch = torch.nn.BatchNorm1d()
        self.flatten = torch.nn.Flatten()

    def forward(self, input):
        x = self.embedding(input)
        x = F.max_pool1d(self.conv1(x), 3, 3, padding='same')
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.batch(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.softmax(self.dense2(x))
        return x

#----------------------------------------------Method2---------------------------------------
class model2(torch.nn.Module):
    def __init__(self):
        super(model2, self).__init__()
        self.embedding = torch.nn.embedding(output_dim=32,
                                            input_dim=2000,
                                            input_length=50)
        self.conv1 = torch.nn.Sequential(torch.nn.Conv1d(256, 3, padding='same', activation='relu'),
                                         torch.nn.MaxPool1d(3, 3, padding='same'))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv1d(32, 3, padding='same', activation='relu'),
                                         torch.nn.Flatten(),
                                         torch.nn.Dropout(0.3),
                                         torch.nn.BatchNorm1d())
        self.dense = torch.nn.Sequential(torch.nn.Linear(32, 256, activation='relu'),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(256, 10, activation='softmax'))

    def forward(self, input):
        x = self.embedding(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense(x)
        return x

#---------------------------------------------------Method3--------------------------------------------------
class model3(torch.nn.Module):
    def __init__(self):
        super(model3, self).__init__()
        self.conv = torch.nn.Sequential()
        self.conv.add_module('embedding', torch.nn.Embedding(output_dim=32, input_dim=2000, input_length=50))
        self.conv.add_module('conv1', torch.nn.Conv1d(256, 3, padding='same', activation='relu'))
        self.conv.add_module('maxpool', torch.nn.MaxPool1d(3, 3, padding='same'))
        self.conv.add_module('conv2', torch.nn.Conv1d(32, 3, padding='same', activation='relu'))
        self.conv.add_module('flatten', torch.nn.Flatten())
        self.conv.add_module('dropout1', torch.nn.Dropout(0.3))
        self.conv.add_module('batch', torch.nn.BatchNorm1d())
        self.conv.add_module('dense1', torch.nn.Linear(32, 256, activation='relu'))
        self.conv.add_module('dropout2', torch.nn.Dropout(0.2))
        self.conv.add_module('dense2', torch.nn.Linear(256, 10, activation='softmax'))

    def forward(self, input):
        x = self.conv(input)
        return x

#-----------------------------------------------------Method4-------------------------------------------------
class model4(torch.nn.Module):
    def __init__(self):
        super(model4, self).__init__()
        self.embedding = torch.nn.embedding(output_dim=32,
                                            input_dim=2000,
                                            input_length=50)
        self.conv = torch.nn.Sequential(
            OrderedDict(['embedding', torch.nn.Embedding(output_dim=32, input_dim=2000, input_length=50),
                         'conv1', torch.nn.Conv1d(256, 3, padding='same', activation='relu'),
                         'maxpool', torch.nn.MaxPool1d(3, 3, padding='same'),
                         'conv2', torch.nn.Conv1d(32, 3, padding='same', activation='relu'),
                         'flatten', torch.nn.Flatten(),
                         'dropout1', torch.nn.Dropout(0.3),
                         'batch', torch.nn.BatchNorm1d(),
                         'dense1', torch.nn.Linear(32, 256, activation='relu'),
                         'dropout2', torch.nn.Dropout(0.2),
                         'dense2', torch.nn.Linear(256, 10, activation='softmax')])
                        )
    def forward(self, input):
        x = self.embedding(input)
        x = self.conv(x)
        return x



