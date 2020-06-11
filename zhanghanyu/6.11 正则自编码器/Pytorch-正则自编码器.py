# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月11日
# 内容：正则自编码器
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------#
# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看解码效果
# 7、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
path = 'D:\\研究生\\代码\\Keras代码\\3.AutoEncoder(自编码器)\\mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 数据放到本地路径

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

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))


#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、构建正则自编码器模型 ---------------------
input_size = 784
hidden_size = 32
output_size = 784
class Norautoencoder(nn.Module):
    def __init__(self):
        super (Norautoencoder,self).__init__()
        self.dense=nn.Sequential(
                   nn.Linear(784,32),
                   nn.ReLU(),
                   nn.Linear(32,784),
                   nn.Sigmoid()
                    )
    def forward(self,x):
        r=self.dense(x)
        return r

'''
x = Input(shape=(input_size,))
h = Dense(hidden_size, activation='relu', activity_regularizer=regularizers.l1(10e-5))(x)
r = Dense(output_size, activation='sigmoid')(h)
'''
autoencoder = Norautoencoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()
Epoch = 15
autoencoder.compile(optimizer='adam', loss='mse')

#  --------------------- 3、构建正则自编码器模型 ---------------------


#  --------------------- 4、训练 ---------------------
# 使用dataloader载入数据，小批量进行迭代
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(X_train, X_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)


loss_list = []

for t in range(5):
    running_loss = 0
    for step, (X_train, X_train) in enumerate(loader):
        train_prediction = autoencoder(X_train)
        loss = loss_func(train_prediction, X_train)  # 计算损失
        loss_list.append(loss)  # 使用append()方法把每一次的loss添加到loss_list中
        regularization_loss = 0
        for param in autoencoder.parameters():
            regularization_loss += torch.sum(torch.abs(param))

        loss = loss + 0.0001 * regularization_loss
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

#  --------------------- 4、训练 ---------------------


