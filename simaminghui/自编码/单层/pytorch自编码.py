# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/29 002913:37
# 文件名称：pytorch自编码
# 开发工具：PyCharm

# 数据路径
import numpy
import torch

path = "D:\DataList\mnist\mnist.npz"
f = numpy.load(path)

x_train, x_test = f['x_train'], f["x_test"]

# 归一化
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# 将矩阵转为向量
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

# 转为tensor格式
x_train = torch.Tensor(x_train)
x_test = torch.Tensor(x_test)

# ----------------------------------构建模型----------------------------------
input_size = 784  # 向量长度
hidden_size = 64
output_size = 784


class EncodeModel(torch.nn.Module):
    def __init__(self):
        super(EncodeModel, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(784, 64),  # 输入在前，输出在后，与keras相反
            torch.nn.ReLU(),
            torch.nn.Linear(64, 784),
            torch.nn.Sigmoid(),  # 输入之前数据进行归一化，数字在0-1之间，所以输出也是0-1之间

        )

    def forward(self, x):
        output = self.dense(x)
        return output


model = EncodeModel()

# 优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()
Epoch = 5

# 数据转换成tensor
# x_train = torch.tensor(x_train)
# x_test = torch.tensor(x_test)
# print(x_train[0])

# 开始训练
import torch.utils.data as Data

torch_dataset = Data.TensorDataset(x_train, x_train)
# 将数据分成批量
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)

loss_list = []

# 开始训练

for i in range(Epoch):
    running_loss = 0
    for step, (x_train, x_train) in enumerate(loader):
        train_prediction = model(x_train)
        loss = loss_func(train_prediction, x_train)  # 得到损失值
        loss_list.append(loss)
        optimizer.zero_grad()  # 清零
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        running_loss += loss.item()
    else:
        print(f"第{i}代训练损失为：{running_loss / len(loader)}")

import matplotlib.pyplot as plt

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.plot(loss_list, 'c-')
plt.show()
