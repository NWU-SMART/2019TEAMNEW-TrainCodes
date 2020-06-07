# ----------------开发者信息--------------------------------#
# 开发者：姜媛
# 开发日期：2020年6月1日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 pytorch, matplotlib, numpy, sklearn 和 panda的包
# 2、参数定义
# 3、招聘数据数据导入
# 4、分词和提取关键词
# 5、建立字典，并使用
# 6、模型构造
# 7、模型训练
# 8、可视化训练
# ----------------------   代码布局： ---------------------- #

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import jieba
import jieba.analyse as analyse
import torch.nn.functional as F
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
#  -------------------------- 导入需要包 ----------------------------------

#  --------------------------2、招聘数据数据导入 -------------------------------
job_detail_pd = pd.read_csv('C:\\Users\\HP\\Desktop\\每周代码学习\\招聘信息文本分类\\job_detail_dataset.csv', encoding='UTF-8') # 数据读取
print(job_detail_pd.head(5))  # 显示前5个数据
label = list(job_detail_pd['PositionType'].unique())  # 将不重复的工作类型列出
print(label)  # 输出标签


# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引，将工作类型转化为数字标签
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset) # 加入label列
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)
#  -------------------------- 2、招聘数据数据导入 -------------------------------

#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))


# 加入新的一列
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)


# 提取关键词 ,topK=50，50个关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))


# 加入新的一列
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
#  --------------------------3、分词和提取关键词 -------------------------------

#  --------------------------4、建立字典，并使用 -------------------------------
# 建立2000个词的字典

token = Tokenizer(num_words=500)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前500的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()

# 转换为tensor形式
x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)
#  -------------------------- 4、建立字典，并使用 -------------------------------

#  -------------------------- 5、class类继承模型 -------------------------------
batch_size = 256
epochs = 5


class Employment(nn.Module):
    def __init__(self):
        super(Employment, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=500, embedding_dim=32)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.flatten = torch.nn.Flatten()  # 平展
        self.linear1 = torch.nn.Linear(1600, 256)
        self.relu1 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.25)
        self.linear2 = torch.nn.Linear(256, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        out = self.softmax(x)
        return out


model = Employment()
print(model)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.2)
#  -------------------------- 5、class类继承模型 -------------------------------

#  -------------------------- 6、模型训练  -------------------------------
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()
print("-----------训练开始-----------")
iteration = []  # list存放epoch数
loss_total = []  # list存放损失
for epoch in range(epochs):
        predict = model(x_train)  # output
        loss_epoch_train = loss_func(predict, y_train)  # cross entropy loss
        iteration.append(epoch)  # 将epoch放到list中
        loss_total.append(loss_epoch_train)  # 将loss放到list中
        optimizer.zero_grad()  # clear gradients for this training step
        loss_epoch_train.backward()  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        print('epoch %3d , loss %3d' % (epoch, loss_epoch_train))
print("-----------训练结束-----------")
torch.save(model.state_dict(), "job information.pkl")  # 保存模型参数
# -------------------------------6、模型训练------------------------

#  -------------------------- 7、模型可视化    ------------------------------
plt.plot(iteration,loss_total, label="Train loss")
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left') # loc:图例位置
plt.show()
#  -------------------------- 7、模型可视化    ------------------------------
