# ----------------开发者信息--------------------------------#
# 开发者：高强
# 开发日期：2020.05.25
# 开发框架：pytorch
# 修改日期：2020.05.26
# 修改内容: 解决Flatten()层报错问题
# 备注：用到Flatten()这个层，得先更新pytorch,不然报错
#-----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 pytorch相关包
# 2、招聘数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
#--------------------------------------------------------------#
#-----------------------------------------招聘数据导入------------------------------------------------#
import pandas as pd
# 加载数据#
job_detail_pd = pd.read_csv('F:\Keras代码学习\keras\keras_datasets\job_detail_dataset.csv',encoding='utf-8')
print(job_detail_pd.head()) # 打印出前五个
# 十类标签：# ['项目管理', '移动开发', '后端开发', '前端开发', '测试', '高端技术职位', '硬件开发', 'dba', '运维', '企业软件']
label = list(job_detail_pd['PositionType'].unique()) # unique()返回参数数组中所有不同的值，并按照从小到大排序
print(label) # 打印标签

# 返回label列表对应值的索引
def label_dataset(row):
    num_label =label.index(row)
    return num_label

job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset) # 对应起来
job_detail_pd = job_detail_pd.dropna() # 删除空行 （默认滤除所有包含NaN）
print(job_detail_pd.head()) # 打印前五个 (带标签)

#----------------------------------------分词和提取关键词----------------------------------#
import jieba
import jieba.analyse as analyse

# 中文分词
def chinese_word_cut(row):
    return "".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
print(job_detail_pd.head())# 打印前五个

# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))

job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
print(job_detail_pd.head())# 打印前五个

#---------------------------------------建立字典，并使用-----------------------------------#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
# 建立2000个词的字典
token = Tokenizer(num_words = 2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])#按单词出现次数排序，排序前2000的单词会列入词典中
# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])
# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
# 训练数据
x_train = Job_Description_Seq_Padding    # 训练数据 数字列表（长度为50）
y_train = job_detail_pd['label'].tolist()# 训练标签转化为列表
import torch
import numpy as np
from torch.autograd import Variable
y_train = np.array(y_train)  # 标签转换为array形式
x_train = Variable(torch.from_numpy(x_train)).long() # x_train变为variable数据类型
y_train = Variable(torch.from_numpy(y_train)).long() # y_train变为variable数据类型

#----------------------------------------------训练模型---------------------------------------------------------#
################################  方法一：自定义class      ############################################
import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self):
#         super(Model,self).__init__()
#         self.Embedding = torch.nn.Embedding(2000,32)
#         self.Dropout1 = torch.nn.Dropout(0.2)
#         self.Flatten = torch.nn.Flatten()
#         self.linear1 = torch.nn.Linear(1600,256)
#         self.relu1 = torch.nn.ReLU()
#         self.Dropout2 = torch.nn.Dropout(0.25)
#         self.linear2 = torch.nn.Linear(256,10)
#         self.softmax = torch.nn.Softmax()
#
#     def forward(self,x):
#         x = self.Embedding(x)
#         x = self.Dropout1(x)
#         x = self.Flatten(x)
#         x = self.linear1(x)
#         x = self.relu1(x)
#         x = self.Dropout2(x)
#         x = self.linear2(x)
#         x = self.softmax(x)
#
#         return x

################################  方法二：Sequential    ############################################

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Embedding(2000,32),
            torch.nn.Dropout(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(1600,256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256,10),
            torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


###############################################################################################################
model = Model()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()
Epoch = 5

# 开始训练 #
for t in range(Epoch):

    x = model(x_train)  # 向前传播
    loss = loss_fn(x, y_train)  # 计算损失
    # 显示损失
    if (t + 1) % 1 == 0:
        print(loss.item())
    # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    optimizer.zero_grad()
    # 计算梯度
    loss.backward()
    # 更新梯度
    optimizer.step()

# # 保存模型
# print("模型保存")
# torch.save(model, '\model.pkl')
# # 加载模型
# print("加载模型")
# model = torch.load('\model.pkl')