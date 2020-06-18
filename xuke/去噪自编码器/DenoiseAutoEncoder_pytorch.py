#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.17
#software：pycharm
#项目名称：去噪自编码器（pytorch）
#--------------         开发者信息--------------------------

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
import torch
import torch.nn as np
import torch.nn as nn
#  -------------------------- 1、导入需要包 -------------------------------
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'D:\\keras_datasets\\mnist.npz'
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

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

# 加入噪声数据

noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  ----------------------------- 3、构建模型 -------------------------------
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding='same'),   # # 1*28*28 --> 32*28*28
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2)    # 32*28*28 --> 32*14*14
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),   # 32*14*14 --> 32*14*14
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2)   # 32*14*14--> 32*7*7
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 32*7*7--> 32*7*7
            torch.nn.ReLU(),
            torch.nn.Upsample((14, 14))     # 32*7*7-> 32*14*14
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),    # 32*14*14--> 32*14*14
            torch.nn.ReLU(),
            torch.nn.Upsample((28, 28))   # 32*28*28--> 32*28*28
        )
        self.output = torch.nn.Sequential(
            torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),    # 32*28*28-> 1*28*28
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # input[60000, 28, 28, 1]---->[6000,1,28,28] 交换位置
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        output = self.output(conv4)
        return output
model = Model()

#  ----------------------------- 3、构建模型 -------------------------------
#---------------------4.训练过程可视化、保存模型----------------------------

print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCELoss()
Epoch = 15
## 开始训练 ##
for t in range(15):
    x = model(x_train_noise)          # 向前传播
    loss = loss_fn(x, x_train_noise)  # 计算损失
    if (t + 1) % 1 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(t + 1, 15, loss.item()))  # 每训练1个epoch，打印一次损失函数的值
    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度
    if (t + 1) % 5 == 0:
        torch.save(model.state_dict(), "./pytorch_NormalizeAutoEncoder_model.pkl")  # 每5个epoch保存一次模型
        print("save model")
#---------------------4.训练过程可视化、保存模型----------------------------