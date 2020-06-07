# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月3日
# 修改日期：
# 修改人：
# 修改内容：使用gpu加速训练，
'''此程序需要在安装了gpu版本的pytorch电脑运行
120行使用了dataloader载入数据，小批量进行迭代
修改了画图方式和损失打印方式
'''

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



#   ---------------------- 数据的载入和处理 ----------------------------
job_detail_pd = pd.read_csv('D:\\Study\\project\\招聘信息文本分类\\job_detail_dataset.csv', encoding='UTF-8')
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


#  -------------------------- 分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))


job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)


# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))


job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)

#  -------------------------- 建立字典 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50  词嵌入前的预处理
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)  # 长度都填充到50
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()  # 把数组转化为列表

x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)


#   ---------------------- 构建模型 ---------------------------
class JobModel(nn.Module):  # 继承torch.nn.Module
    def __init__(self):  # 绑定两个属性
        super(JobModel, self).__init__()
        self.dense =torch.nn.Sequential(torch.nn.Embedding(num_embeddings=2000, embedding_dim=32),
                                        torch.nn.Conv1d(in_channels=50, out_channels=256, kernel_size=3, padding=1, stride=1),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool1d(kernel_size=3, padding=1),
                                        torch.nn.Conv1d(in_channels=256, out_channels=50, kernel_size=3, padding=1, stride=1),
                                        torch.nn.Flatten(),
                                        torch.nn.BatchNorm1d(550),
                                        torch.nn.Dropout(0.2),
                                        torch.nn.Linear(550, 256),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(256, 10),
                                        torch.nn.Softmax()
                                        )

    def forward(self, x):
        x = self.dense(x)

        return x


model = JobModel()  # 实例化招聘模型
print(model)  # 打印模型结构

# 调用GPU加速训练，也就是在模型，x_train后面加上cuda()
model = model.cuda()
x_train = x_train.cuda()
y_train = y_train.cuda()


optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
loss_func = torch.nn.CrossEntropyLoss()


#使用dataloader载入数据，小批量进行迭代，
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=512, shuffle=True, num_workers=0)

#   ---------------------- 训练模型 ---------------------------
loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (x_train, y_train) in enumerate(loader):
        train_prediction = model(x_train)
        loss = loss_func(train_prediction, y_train)  # 计算损失
        loss_list.append(loss) # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        running_loss += loss.item()
    else:
        print(f"第{t}代训练损失为：{running_loss/len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
