# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.16
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# 有问题还需改正
# /------------------ 开发者信息 --------------------*/

# /------------------ 网络层设置 --------------------*/
import torch
from torch.nn import LSTM,Linear,Sequential,Embedding,Softmax
embedding_size = 10000
class net1(torch.nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.text = Sequential(
            Embedding(embedding_size,1000),
            LSTM(1000,64),
            Linear(in_features=64,out_features=32)
        )
        self.question = Sequential(
            Embedding(embedding_size, 1000),
            LSTM(1000,64),
            Linear(in_features=64,out_features=32),
        )
        self.result = Sequential(
            Linear(in_features=32,out_features=64),
            Softmax()
        )
    def forward(self,x,y):
        text_encoder = self.text(x)
        question_encoder = self.question(y)
        feature_all = torch.cat([text_encoder,question_encoder],dim=-1)
        result = self.result(feature_all)
        return feature_all
net = net1()
print(net)
# /------------------ 网络层设置 --------------------*/
num_samples = 1000
max_length = 100
answer_size = 500
import numpy as np
from keras.utils import to_categorical
# 伪造数据
text = np.random.randint(1, embedding_size, size=(num_samples, max_length))
question = np.random.randint(1, embedding_size, size=(num_samples, max_length))
# 随机生成结果并将结果进行one-hot编码
answers = np.random.randint(answer_size, size=num_samples)
answers = to_categorical(answers, answer_size) # one-hot化
text = torch.LongTensor(text)
question = torch.LongTensor(question)
answers = torch.LongTensor(answers)
optim = torch.optim.Adam(net.parameters(),lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()
epoch = 10
for i in range(epoch):
    result = net(text,question)
    loss = loss_fn(result,result)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(i,loss.item())
