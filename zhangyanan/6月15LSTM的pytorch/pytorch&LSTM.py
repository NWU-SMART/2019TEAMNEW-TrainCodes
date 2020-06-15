



# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月12日
# 修改日期：
# 修改人：
# 修改内容：
'''
#  有问题 还在改
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import keras
import torch.nn.functional as F
plt.style.use('ggplot') # 画的更好看
#载入imdb电影评论数据集
data=keras.datasets.imdb
#定义规模
max_word=1000

(x_train,y_train),(x_test,y_test)=data.load_data(num_words=max_word)  # 最多只考虑10000个数据

print(x_train.shape,y_train.shape)
print(x_train[0])  # 评论是已经处理好的数据，不同的数字代表不同的单词
print(y_train[0])  # 正面评价或负面评价 0或1




# 把所有评论都填充到同样长度
max_len=200
x_train=keras.preprocessing.sequence.pad_sequences(x_train,maxlen=max_len)  # 填充到200
x_test=keras.preprocessing.sequence.pad_sequences(x_test,maxlen=max_len)

x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)

import torch.utils.data as Data
torch_dataset = Data.TensorDataset(x_train, y_train)

print(x_train.shape)
print(y_train.shape)
#x_train = x_train.reshape(-1, 25000, 200)
#y_train = y_train.reshape(-1, -1, 25000)
print(x_train.shape)
print(y_train.shape)

class MovieModel(nn.Module):
    def __init__(self):  # 绑定两个属性
        super(MovieModel, self).__init__()
        self.embedding = torch.nn.Embedding(1000, 16)
        #self.lstm = torch.nn.LSTM(16,128,1)
        self.out = torch.nn.Linear(128, 1),
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.embedding(x)
        #x = self.lstm(x)
        x = self.out(x)
        x = self.softmax(x)

        return x



model = MovieModel()  # 实例化招聘模型
#model = model.cuda()
#x_train = x_train.cuda()
#y_train = y_train.cuda()


print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()


#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来

loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)

#   ---------------------- 训练模型 --------------------------
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
print(model)  # 打印模型结构