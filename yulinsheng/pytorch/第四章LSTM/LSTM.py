# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.6
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# /------------------ 程序构成 --------------------*/
'''
1.导入需要的包
2.读取数据
3.数据预处理
4.建立模型
5.训练模型
6.结果显示
7.模型保存和预测
'''
# /------------------ 程序构成 --------------------*/
# /------------------导入需要的包--------------------*/
import numpy as np
# /------------------数据预处理--------------------*/
# 生成随机数种子保证每次结果的代码相同
np.random.seed(10)
# 定义一个简单的数据
data = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
# 将data编码
# 遍历data并将索引写入生成新的数据字典类型
char_to_int = dict((c, i) for i, c in enumerate(data))
int_to_char = dict((i, c) for i, c in enumerate(data))
print(char_to_int)
print(int_to_char)
# 构建数据
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(data) - seq_length, 1):
    # 定义前一个输入
    seq_in = data[i:i + seq_length]
    # 根据前一个的输入定义输出
    seq_out = data[i + seq_length]
    # 前一个数据的索引写入0-1,1-2这种的0写入dataX,1写入dataY
    dataX.append([char_to_int[char] for char in seq_in])
    print(dataX)
    # 后一个数据的索引写入
    dataY.append(char_to_int[seq_out])
    print(dataY)
    print(seq_in, '->', seq_out)
input = np.reshape(dataX,(len(dataX),1,1))
import torch
# 将input归一化
input = input/float(len(data))
input = torch.FloatTensor(input)
# 类别进行one-hot编码

y = torch.LongTensor(dataY)
# 不需要转换为one-hot类型，函数内部会自己处理成 one hot 格式
# y = torch.nn.functional.one_hot(dataY)
# /------------------模型定义--------------------*/
from torch.nn import LSTM,Linear,Softmax
class lstm(torch.nn.Module):
    def __init__(self):
        super(lstm,self).__init__()
        self.input = LSTM(input_size=1,hidden_size=32)
        self.out = Linear(out_features=26,in_features=32)
        self.softmax = Softmax()
    def forward(self,x):
        r_out, _ = self.input(x)
        r_out = r_out[:, -1, :]#表示取序列中的最后一个数据，这个数据长度为hidden_dim
        x = self.out(r_out)
        out = self.softmax(x)
        return out

model = lstm()
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
# /------------------模型定义--------------------*/

# /------------------ 模型训练--------------------*/
import numpy
epoch = 500
for i in range(epoch):
    result = model(input)
    # result = result.squeeze()
    loss = loss_fn(result,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(i,loss.item())
# 结果预测（不是很准）
test = np.reshape([[3]], (1, 1, 1))
test = torch.Tensor(test)
test_result = model(test)
# 找到其中最大的索引
index = np.argmax(test_result.detach().numpy())
# 最大索引对应的值
result = int_to_char[index]
print(result)
