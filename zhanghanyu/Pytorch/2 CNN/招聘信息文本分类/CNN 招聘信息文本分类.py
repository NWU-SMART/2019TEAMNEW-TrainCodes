# ----------------开发者信息----------------------------
# 开发者：张涵毓
# 开发日期：2020年6月3日
# 内容：2.1 CNN-招聘信息文本分类
# 修改内容：
# 修改者：
# ----------------开发者信息----------------------------
# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
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
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
path='D:\\研究生\\代码\\Keras代码\\1.Multi-Layer perceptron(MLP 多层感知器)\\job_detail_dataset.csv'
job_detail_pd = pd.read_csv(path, encoding='UTF-8')
print(job_detail_pd.head(5))
label = list(job_detail_pd['PositionType'].unique())  # 标签
print(label)


# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行
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

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  -------------------------- 5、训练模型 -------------------------------
class CNNic(nn.module):
    def __init__(self):
        super(CNNic,self).__init__()
        self.dense=nn.Sequential(nn.Embedding(num_embeddings=2000,embedding_dim=32),
                                 nn.Conv1d(in_channels=50,out_channels=256,kernel_size=3,padding=1),
                                 nn.ReLU(),
                                 nn.MaxPool1d(kernel_size=3,padding=1),
                                 nn.Conv1d(in_channels=256,out_channels=32,kernel_size=3),
                                 nn.Flatten(),
                                 nn.Dropout(0.3),
                                 nn.BatchNorm1d(550),
                                 nn.Linear(550,out_features=256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, out_features=10),
                                 nn.Softmax()
                                 )
    def forward(self, x):
        x = self.dense(x)
        return x


model = CNNic()  # 实例化招聘模型
print(model)  # 打印模型结构
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)  # 定义优化器为SGD，学习率是1e-3
loss_func = torch.nn.CrossEntropyLoss()   # 定义损失函数为均方误差

batch_size = 256
epochs = 5

optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

print("-----------训练开始-----------")
iteration = []  # list存放epoch数
loss_total = []  # list存放损失
for epoch in range(epochs):
        # train_loss = 0.0
        model.train()  # 训练模式
        predict = model(x_train)  # output
        loss_epoch_train = loss_func(predict,y_train)  # cross entropy loss
        iteration.append(epoch)  # 将epoch放到list中
        loss_total.append(loss_epoch_train)  # 将loss放到list中
        optimizer.zero_grad()  # clear gradients for this training step
        loss_epoch_train.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        print('epoch %3d , loss %3d' % (epoch, loss_epoch_train))
print("-----------训练结束-----------")
torch.save(model.state_dict(), "job information.pkl")  # 保存模型参数
# -------------------------------模型训练------------------------

#  -------------------------- 6、模型可视化    ------------------------------
plt.plot(iteration,loss_total, label="Train loss")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left') # loc:图例位置
plt.show()
