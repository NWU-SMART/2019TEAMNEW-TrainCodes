# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年5月21日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import jieba
import jieba.analyse as analyse
from torch.autograd import Variable
import torch.nn as nn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
#  -------------------------- 1、导入需要包 -------------------------------

#  -------------------------- 2、数据载入与预处理 -------------------------------
job_detail_pd = pd.read_csv('data.csv', encoding='UTF-8') # 导入数据
label = list(job_detail_pd['PositionType'].unique())# 将职位设置为标签，unique函数为去除重复项

def label_dataset(row):
     num_label = label.index(row) #按照label出现次序编号
     return num_label
job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset) #将编号添加到数据中
job_detail_pd = job_detail_pd.dropna() # 删除空行

def chinese_word_cut(row):
     return " ".join(jieba.cut(row))  #使用jieba将句子分割成词汇
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)# 将分割的词汇添加到数据中

def key_word_extract(texts):
  return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
# (待提取的文本,返回多少权重最大的关键词,是否一并返回关键词权重值，是否筛选)
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract) # 将关键词添加到数据中


token = Tokenizer(num_words = 100) # 设置一个100词的字典
token.fit_on_texts(job_detail_pd['Job_Description_key_word']) #按单词出现次数排序，排序前100的单词会存入字典中
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)# 所有“数字列表”长度都是50
#  -------------------------- 2、数据载入与预处理 -------------------------------

#  -------------------------- 3、训练数据类型变换 -------------------------------
x_train = Job_Description_Seq_Padding    #数组形式，每句话固定50个词，这50个词经过编码，变成了50个数字（0-100），不够50个数字的补上0
y_train = job_detail_pd['label'].tolist()# 标签转换为列表形式
y_train = np.array(y_train)  # 标签转换为array形式
x = Variable(torch.from_numpy(x_train)).long() # x_train变为variable数据类型
y = Variable(torch.from_numpy(y_train)).long() # y_train变为variable数据类型
#  -------------------------- 3、训练数据类型变换 -------------------------------

#  -------------------------- 4、模型训练以及保存   --------------------------------
class TextNet(nn.Module):
    def __init__(self):
        super(TextNet, self).__init__()
        self.linear1 = torch.nn.Embedding(100,32) # embedding层，100个词，词向量32
        self.linear2 = nn.Sequential(
                        nn.Conv1d(32,256,kernel_size=3,padding=1),# 一维卷积
                        nn.ReLU(),   #激励函数层ReLU
                        nn.MaxPool1d(3,padding=1))#池化层
        self.linear3 = nn.Sequential(
                        nn.Conv1d(256,32,kernel_size=3,padding=1),# 一维卷积
                        nn.ReLU()) #激励函数层ReLU
        self.linear4 = torch.nn.BatchNorm1d(32) #批标准化
        self.linear5 = torch.nn.Flatten() #平展
        self.linear6 = torch.nn.Dropout(0.3) 
        self.linear7 = torch.nn.Linear(544, 256)# 第一层全连接
        self.linear8 = torch.nn.Dropout(0.2)
        self.linear9 = torch.nn.ReLU() #激励函数层ReLU
        self.linear10 = torch.nn.Linear(256, 4)# 第二层全连接
        self.linear11 = torch.nn.Softmax()#激励函数层softmax
    def forward(self, x):
        x = self.linear1(x)# 输入x经过第一层
        x =x.permute(0,2,1) # 需要对输入调整一下参数，使得通道在第二个
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        x = self.linear7(x)
        x = self.linear8(x)
        x = self.linear9(x)
        x = self.linear10(x)
        y_pred = self.linear11(x)# 经过激活函数
        return y_pred #预测返回结果
model = TextNet()# 定义model
loss_fn = nn.CrossEntropyLoss() #交叉熵损失函数
learning_rate = 1e-4 # 学习率
EPOCH = 5  # epoch,迭代多少次
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #SGD优化器
for i in range(EPOCH):
    # 向前传播
    y_pred= model(x)
    # 计算损失
    loss = loss_fn(y_pred, y)
    # 梯度清零
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 更新梯度
    optimizer.step()

    if (i+1) % 1 == 0:#每训练1个epoch，打印一次损失函数的值
        print(loss.data)
    if (i + 1) % 5 == 0: #每训练5个epoch,保存一次模型
        torch.save(model.state_dict(), "./model.pkl")  # 保存模型
        print("save model")
#  -------------------------- 4、模型训练以及保存   --------------------------------
#  -------------------------- 5、加载模型并预测    ------------------------------
model.load_state_dict(torch.load("./model.pkl",map_location=lambda storage, loc: storage)) # 加载训练模型
print("load model")
y_new =model(x[0].reshape(1,50))#通过训练模型预测值，输入第一个x,得到第一个预测y
print(list(y_new[0]).index(max(y_new[0])))# 输出第一个预测y
print(y_train) # 输出真实的y
#  -------------------------- 5、加载模型并预测    ------------------------------
