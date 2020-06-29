# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.6.27
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： /
# /------------------ 开发者信息 --------------------*/

# 单个模型建立
import torch
embedding_size = 10000
from torch.nn import Module,LSTM,Linear,Sequential,\
    Embedding,Conv2d,MaxPool2d,ReLU,Softmax
class mimo(Module):
    def __init__(self):
        super(mimo,self).__init__()
        self.lstm = Sequential(
            # Embedding(embedding_size, 1000),
            LSTM(1000, 64),
            Linear(in_features=64, out_features=32)
        )
        # 假设输入28*28*3
        self.cnn = Sequential(
            Conv2d(3,16,kernel_size=3,padding=1,stride=1),#28*28*16
            ReLU(),
            MaxPool2d(kernel_size=2),#14*14*16

            Conv2d(16,32,kernel_size=3,padding=1,stride=1),#14*14*32
            ReLU(),
            MaxPool2d(kernel_size=2),#7*7*32

            Conv2d(32,64,kernel_size=3,padding=1,stride=1),#4*4*32
            ReLU(),

            Linear(out_features=32,in_features=512) #将图片转换为32维的
        )
        self.mult = Sequential(
            Linear(in_features=32,out_features=8),
            ReLU(),
            Linear(in_features=8,out_features=1),
            Softmax()
        )


    def forward(self, x,y):
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)
        text_input = self.lstm(x)
        image_input = self.cnn(y)
        multi_fea = torch.cat([text_input,image_input],0)
        result = self.mult(multi_fea)
        return result
model = mimo()
print(model)

class loss_me(Module):
    def __init__(self):
        super(loss_me,self).__init__()
    def forward(self,y_pred,y_test):
        loss = torch.mean(torch.abs((y_pred**2-y_test**2)))
        return loss
optim = torch.optim.Adam(model.parameters(),lr=1e-4)
loss = loss_me()
