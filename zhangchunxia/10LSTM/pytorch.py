# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月18日
# 内容:LSTM实现词性的简单标注
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入pytorch的包
# 2、数据处理
# 3、建立模型
# 4、训练模型
# ----------------------   代码布局： -------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、数据准备 -------------------------------------
'''
循环神经网络是维持某种状态的网络。例如，它的输出可以用作下一个输入的一部分
这样当网络通过序列时，信息可以传播。在LSTM的情况下，对于序列中的每个元素
存在相应的隐藏状态ht，其原则上可以包含来自序列中较早的任意点的信息
可以使用隐藏状态来预测语言模型中的单词，词性标签以及无数其他内容
'''
torch.manual_seed(1)#在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果，
# 当得到比较好的结果时我们通常希望这个结果是可以复现的，在pytorch中，通过设置随机数种子也可以达到这个目的。
#数据准备
#然后为了将数据放到网络里面，需要做一个编码单词的函数
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),# DET限定词，NN名词，V动词
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}#创建空字典
for sent, tags in training_data:# 将train_data中的词按顺序编号
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)# 输出带有编号的train_data中出现的词
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}#给词性编号
EMBEDDING_DIM = 6 #词向量的维度
HIDDEN_DIM = 6 #隐藏层的单元数
#  ---------------------  3、构建模型 -----------------------------------
class lstmm(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size):
        super(lstmm,self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)#nn.Embedding是pytorch内置的词嵌入工具
        #vocab_size是参数词库中的单词数，embedding_size是词嵌入表示成的维度
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)#LSTM层，第一个参数是输入的词向量维度，第二个是隐藏层的单元数
        self.hidden2tag = nn.Linear(hidden_dim,tagset_size)#线性层
    def forward(self,sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))#LSTM层
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))#线性层
        tag_scores = F.log_softmax(tag_space,dim=1)#这里用log_softmax是因为下面的损失函数用的是NLLLoss() 即负对数似然损失
        return tag_scores
#前向传播的过程，首先词嵌入（将词表示为向量），然后通过LSTM层，线性层，最后通过一个logsoftmax函数
#输出结果，用于多分类
#  ---------------------  3、构建模型 -----------------------------------
#  ---------------------- 4、模型训练 -----------------------------------
model =  lstmm(EMBEDDING_DIM,HIDDEN_DIM,len(word_to_ix),len(tag_to_ix))
loss_fn = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
for i in range(30):
    for sentence,tags in training_data:
        #获得输入
        sentence_in = prepare_sequence(sentence,word_to_ix)#训练集x_train
        targets = prepare_sequence(tags,tag_to_ix)#训练集y_train
        #前向传播
        tag_scores = model(sentence_in)#跑网络
        #计算损失
        loss = loss_fn(tag_scores, targets)
        model.zero_grad()  # 梯度清零
        #后向传播
        loss.backward()
        #更新参数
        optimizer.step()#更新参数
        print('i [{}/{}], loss:{:.4f}'.format(i + 1, 30, loss.item()))
#测试过程
with torch.no_grad():
   inputs = prepare_sequence(training_data[0][0],word_to_ix)
   tag_scores=model(inputs)
#  ---------------------- 4、模型训练 -----------------------------------
