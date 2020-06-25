# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/24 002422:11
# 文件名称：__init__.py
# 开发工具：PyCharm

import pandas
import torch

# 将得到DataFrame格式转为矩阵
x_train = pandas.read_csv('D:\DataList\job\dataset_x_train.csv').values
y_train = pandas.read_csv('D:\DataList\job\dataset_y_train.csv').values.reshape(len(x_train)).tolist()

# 数据讲过torch处理
x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)

print(x_train, len(x_train))
print(y_train,len(y_train))


#   ---------------------- 构建模型 ---------------------------
class JobModel(torch.nn.Module):
    def __init__(self):
        super(JobModel, self).__init__()
        # 定义全连接
        self.dense = torch.nn.Sequential(
            torch.nn.Embedding(2000, 32, max_norm=50), # 将维
            torch.nn.Dropout(0.2),
            torch.nn.Flatten(), # 多维的输入一维化，进行全连接层
            torch.nn.Linear(1600, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 10),
            torch.nn.Softmax()  # 输出为一个概率
        )

    def forward(self, x):
        x = self.dense(x)
        return x


# 模型实例化
model = JobModel()

# 定义优化函数和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) # 随机梯度下降
loss_func = torch.nn.CrossEntropyLoss() # 交叉熵

# -----------------------------------训练模型-------------------------------
loss_list = []  # 存放每次的损失

# 开始训练
for i in range(5):
    y_pre = model(x_train)
    loss_list.append(loss_func(y_pre, y_train))
    optimizer.zero_grad()
    loss_list[i].backward()
    optimizer.step()

print(loss_list)


import matplotlib.pyplot
matplotlib.pyplot.plot(loss_list,'r')
matplotlib.pyplot.show()