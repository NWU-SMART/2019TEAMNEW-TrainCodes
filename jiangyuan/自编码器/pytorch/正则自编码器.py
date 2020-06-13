# ----------------开发者信息--------------------------------
# 开发者：姜媛
# 开发日期：2020年6月12日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from keras.layers import Dense, Input
from keras.models import Model
import torch.utils.data as Data
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------
path = 'C:\\Users\\HP\\Desktop\\每周代码学习\\自编码器\\mnist.npz'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
# --------- 输出语句结果 --------

# 数据准备
# 转为Tensor格式
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)

X_train = X_train.permute(0, 3, 2, 1)
X_test = X_test.permute(0, 3, 2, 1)
# 使用dateloder载入数据
torch_dataset = Data.TensorDataset(X_train, X_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、构建正则自编码器模型 ---------------------
input_size = 784
hidden_size = 32
output_size = 784


class Norautoencoder(nn.Module):
    def __init__(self):
        super(Norautoencoder, self).__init__()
        self.dense = nn.Sequential(nn.Linear(784, 32),
                                   nn.ReLU(),
                                   nn.Linear(32, 784),
                                   nn.Sigmoid()
                                   )

    def forward(self, x):
        output = self.dense(x)
        return output
#  --------------------- 3、构建正则自编码器模型 ---------------------


#  --------------------- 4、模型训练 ---------------------
autoencoder = Norautoencoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()
autoencoder.compile(optimizer='adam', loss='mse')

# 使用GPU训练
model = autoencoder.cuda()
X_train = X_train.cuda()

# 设定peochs和batch_size大小
epochs = 5
batch_size = 128
# 训练模型
history = Norautoencoder.fit(X_train, X_train, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_data=(X_test, X_test)
                             )

loss_list = []  # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (X_train, X_train) in enumerate(loader):
        train_prediction = model(X_train)
        loss = loss_func(train_prediction, X_train)  # 计算损失
        loss_list.append(loss)  # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch后清零梯度
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        running_loss += loss.item()
    else:
        print(f"第{t}代训练损失为：{running_loss/len(loader)}")  # 打印平均损失
#  --------------------- 4、模型训练 ---------------------


#  ---------------------5、损失可视化 ---------------------
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
#  ---------------------5、损失可视化 ---------------------
