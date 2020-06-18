# ----------------开发者信息--------------------------------#
# 开发者：张涵毓
# 开发日期：2020年6月12日
# 内容：去噪自编码器
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
# 本地读取数据 (本地路径)
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


#  --------------------- 3、构建去噪自编码器模型 ---------------------
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        # 编码器
        self.encoder=nn.Sequential(nn.Conv2d(1,32,3,padding='same'),
                                   nn.ReLU(),# 28*28*1 --> 28*28*32
                                   nn.MaxPool2d(2,padding='same'), # 28*28*32 --> 14*14*32
                                   nn.Conv2d(32,32,2,padding='same'),
                                   nn.ReLU(),# 14*14*32 --> 14*14*32
                                   nn.MaxPool2d(2,padding='same')
                                   ) # 14*14*32 --> 7*7*32
        #解码器
        self.decoder=nn.Sequential(nn.Conv2d(32,32,3,padding='same'),
                                   nn.ReLU(), # 7*7*32 --> 7*7*32
                                   nn.Upsample(2), # 7*7*32 --> 14*14*32
                                   nn.Conv2d(32,32,3,padding='same'),
                                   nn.ReLU(),# 14*14*32 --> 14*14*32
                                   nn.Upsample(2),# 14*14*32 --> 28*28*32
                                   nn.Conv2d(32,1,3,padding='same'),# 28*28*32 --> 28*28*1
                                   nn.Sigmoid()
                                   )
    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

autoencoder = autoencoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()
Epoch = 5
# 使用dataloader载入数据，小批量进行迭代
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(X_train_noisy, X_train_noisy)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)


loss_list = []
for epoch in range(5):
    running_loss = 0
    for step, (X_train_noisy, X_train_noisy) in enumerate(loader):
        train_prediction = autoencoder(X_train_noisy)
        loss = loss_func(train_prediction, X_train_noisy)  # 计算损失
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

