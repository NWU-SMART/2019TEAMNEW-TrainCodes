#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.1
#software：pycharm
#项目名称：招聘信息（pytorch）


# ----------------------   代码布局： ----------------------
# 1、导入包
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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 文件放置在目录  D:\keras\招聘信息\job_detail_dataset.csv
job_detail_pd = pd.read_csv('job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))                          # 打印前五行 #
label = list(job_detail_pd['PositionType'].unique())  # 标签 #
print(label)

# 为工作描述设置标签的id #
def label_dataset(row):
    num_label = label.index(row)        # 返回label列表对应值的索引 #
    return num_label
job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行 #
job_detail_pd.head(5)
#  -------------------------- 2、招聘数据数据导入 -------------------------------


#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)

# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
#  -------------------------- 3、分词和提取关键词 -------------------------------


#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50  词嵌入前的预处理
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)  # 长度都填充到50
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()  # 把数组转化为列表
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  ------------------------------- 5、模型建立   ----------------------------------

#  ------------------------- 5.1 继承add_module类   ---------------------------#
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.input = torch.nn.Sequential()
        self.input.add_module('embedding', torch.nn.Embedding(output_dim=32, input_dim=2000, input_length=50))
        self.input.add_module('conv1', torch.nn.Conv1d(256, 3))
        self.input.add_module('relu1', torch.nn.ReLU())
        self.input.add_module('maxpool', torch.nn.MaxPool(3, 3))
        self.input.add_module('conv2', torch.nn.Conv1d(32, 3))
        self.input.add_module('relu2', torch.nn.ReLU())
        self.input.add_module('flatten', torch.nn.Flatten())
        self.input.add_module('dropout1', torch.nn.Dropout(0.3))
        self.input.add_module('dense1', torch.nn.Linear(32, 256))
        self.input.add_module('relu3', torch.nn.ReLU())
        self.input.add_module('dropout2', torch.nn.Dropout(0.2))
        self.input.add_module('dense2', torch.nn.Linear(256, 10, activation='softmax'))

    def forward(self, input):
        x = self.conv(input)
        return x

#  --------------------- 5.2封装 torch.nn.Sequential()  ----------------------#
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Sequential(torch.nn.Embedding(2000,32),  #输入输出大小
                                         torch.nn.Dropout(0.2),        #丢弃20%
                                         torch.nn.Flatten(),           # 平展
                                         torch.nn.Linear(1600,256),
                                         torch.nn.ReLU(),              #relu激活
                                         torch.nn.Dropout(0.25),
                                         torch.nn.Linear(256,10),
                                         torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


#  ------------------------- 5.3 OrderedDict子类   ---------------------------#
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.layer = nn.Sequential(OrderedDict([('embedding', torch.nn.Embedding(output_dim=32, input_dim=2000, input_length=50))
                                                ('conv1', torch.nn.Conv1d(256, 3))
                                                ('relu1', torch.nn.ReLU())
                                                ('maxpool', torch.nn.MaxPool(3, 3))
                                                ('conv2', torch.nn.Conv1d(32, 3))
                                                ('relu2', torch.nn.ReLU())
                                                ('flatten', torch.nn.Flatten())
                                                ('dropout1', torch.nn.Dropout(0.3))
                                                ('dense1', torch.nn.Linear(32, 256))
                                                ('relu3', torch.nn.ReLU())
                                                ('dropout2', torch.nn.Dropout(0.2))
                                                ('dense2', torch.nn.Linear(256, 10, activation='softmax'))])
    def forward(self, x):
        x = self.dense(x)
        return x
#  ------------------------------ 5.4 类继承   -----------------------------#
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.embedding = torch.nn.embedding(output_dim=32,input_dim=2000,input_length=50)
        self.conv1 = torch.nn.Sequential(torch.nn.Conv1d(256, 3, padding='same', activation='relu'),
                                         torch.nn.MaxPool(3, 3, padding='same'))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv1d(32, 3, padding='same', activation='relu'),
                                         torch.nn.Flatten(),
                                         torch.nn.Dropout(0.3))
        self.dense = torch.nn.Sequential(torch.nn.Linear(32, 256, activation='relu'),
                                         torch.nn.Dropout(0.2),
                                         torch.nn.Linear(256, 10, activation='softmax'))

    def forward(self, input):
        x = self.embedding(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dense(x)
        return x