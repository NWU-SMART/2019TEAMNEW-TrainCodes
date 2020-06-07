# ----------------开发者信息---------------------------------------------------------------------
# 开发者：刘盱衡
# 开发日期：2020年5月29日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息---------------------------------------------------------------------

#  -------------------------- 1、导入需要包 -----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#  -------------------------- 1、导入需要包 -----------------------------------------------------

#  --------------------- 2、获取数据及与数据预处理 ----------------------------------------------
torch.manual_seed(1)# 生成随机数种子

#准备数据的阶段，获取编号
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# DET限定词，NN名词，V动词
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}#创建空字典

# 将train_data中的词按顺序编号
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)# 输出带有编号的train_data中出现的词
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}#给词性编号

EMBEDDING_DIM = 6 #词向量的维度
HIDDEN_DIM = 6 #隐藏层的单元数
#  --------------------- 2、获取数据及与数据预处理 -----------------------------------------------

#  --------------------- 3、LSTM建模 -------------------------------------------------------------
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        #nn.Embedding(vocab_size, embedding_dim) 是pytorch内置的词嵌入工具
        #第一个参数词库中的单词数,第二个参数将词向量表示成的维数

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        #LSTM层，nn.LSTM(arg1, arg2) 第一个参数输入的词向量维数，第二个参数隐藏层的单元数

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)# 线性层

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) #将词表示成向量
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))# LSTM层
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) # 线性层
        tag_scores = F.log_softmax(tag_space, dim=1) #,通过一个logsoftmax函数,输出结果
        return tag_scores
#  --------------------- 3、LSTM建模 --------------------------------------------------------------

#  -------------------------- 4、训练模型   -------------------------------------------------------
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))# 导入模型
loss_function = nn.NLLLoss() # 损失函数
optimizer = optim.SGD(model.parameters(), lr=0.1) #优化器
for epoch in range(30):  
    for sentence, tags in training_data:   
        sentence_in = prepare_sequence(sentence, word_to_ix)# x_train
        targets = prepare_sequence(tags, tag_to_ix)# y_train
        tag_scores = model(sentence_in)#前向传播
        loss = loss_function(tag_scores, targets)#计算损失
        model.zero_grad()#梯度清零
        loss.backward()#后向传播
        optimizer.step()#更新参数
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 30, loss.item()))
        #每训练1个epoch，打印一次损失函数的值
#  -------------------------- 4、训练模型   -------------------------------------------------------

