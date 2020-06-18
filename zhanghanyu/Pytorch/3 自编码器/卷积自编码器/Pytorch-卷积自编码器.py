# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月10日
# 内容：卷积自编码器-Pytorch
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
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# 'D:\\研究生\\代码\\Keras代码\\3.AutoEncoder(3 自编码器)\\mnist.npz'(本地路径)
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

# 数据格式进行转换
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#  数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
# 输出X_train和X_test维度
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28, 1)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------


#  --------------------- 3、构建卷积自编码器模型 ---------------------

# 输入维度为 1*28*28
#x = Input(shape=(28, 28,1))
class CNNautoencoder(nn.Module):
    def __init__(self):
        super(CNNautoencoder,self).__init__()
    #编码器
        self.encoder=nn.Sequential(
            nn.Conv2d(1,16,3,padding='same'),
            nn.ReLU(),# 1*28*28 --> 16*28*28
            nn.MaxPool2d(2,padding='same'), # 16*28*28 --> 16*14*14
            nn.Conv2d(16,8,3,padding='same'),
            nn.ReLU(), # 16*14*14 --> 8*14*14
            nn.MaxPool2d(2,padding='same'),# 8*14*14 --> 8*7*7
            nn.Conv2d(8,8,3,padding='same'),
            nn.ReLU(),# 8*7*7 --> 8*7*7
            nn.MaxPool2d(2,padding='same')# 8*7*7 --> 8*4*4
        )
    #解码器
        self.decoder=nn.Sequential(
            nn.Conv2d(8,8,3,padding='same'),
            nn.ReLU(),# 8*4*4 --> 8*4*4
            nn.Upsample(2),# 8*4*4 --> 8*8*8
            nn.Conv2d(8,8,3,padding='same'),
            nn.ReLU(),# 8*8*8 --> 8*8*8
            nn.Upsample(2), # 8*8*8 --> 8*16*16
            nn.Conv2d(8,16,4),
            nn.ReLU(),# 8*16*16 --> 16*14*14 (not same)
            nn.Upsample(2),# 16*14*14 --> 16*28*28
            nn.Conv2d(16,1,3,padding='same'),
            nn.Sigmoid()
        )
    def forward(self,x):
        enc=self.encoder(x)
        dec=self.decoder(enc)
        return dec
model = CNNautoencoder()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()
Epoch = 5
#  --------------------- 3、构建卷积自编码器模型 ---------------------

# 使用dataloader载入数据，小批量进行迭代
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(X_train, X_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)


loss_list = []
for epoch in range(5):
    running_loss = 0
    for step, (X_train, X_train) in enumerate(loader):
        train_prediction = model(X_train)
        loss = loss_func(train_prediction, X_train)  # 计算损失
        loss_list.append(loss)  # 使用append()方法把每一次的loss添加到loss_list中

        optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        #print(loss)
        running_loss += loss.item()
    else:
        print(f"第{epoch}代训练损失为：{running_loss/len(loader)}")

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()