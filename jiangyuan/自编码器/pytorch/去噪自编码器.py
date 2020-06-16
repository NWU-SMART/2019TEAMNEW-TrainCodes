# ----------------开发者信息--------------------------------
# 开发者：姜媛
# 开发日期：2020年6月15日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息--------------------------------


#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#  -------------------------- 1、导入需要包 -------------------------------


#  --------------------- 2、读取手写数据集并对图像预处理 ---------------------
path = 'C:\\Users\\HP\\Desktop\\每周代码学习\\单层自编码器\\mnist.npz'
f = np.load(path)
# 以npz结尾的数据集是压缩文件，里面还有其他的文件
# 使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()
# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
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
# 加入噪声数据

noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

X_train_noise = torch.Tensor(X_train_noisy)  # 转换为tenser
X_test_noise = torch.Tensor(X_test_noisy)    # 转换为tenser

X_train_noise =X_train_noise.permute(0, 3, 2, 1)
X_test_noise =X_test_noise.permute(0, 3, 2, 1)
#  --------------------- 2、读取手写数据集并对图像预处理 ---------------------


#  --------------------- 3、构建去噪自编码器模型 --------------------
class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.encoder = nn.Sequential(                  # 编码层
            nn.Conv2d(1, 16, 3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(                  # 定义解码层
            nn.ConvTranspose2d(8, 16, 3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        encode = self.encoder(x)  # 编码层
        decode = self.decoder(encode)  # 解码层
        return decode


model = EncoderModel()
# 调用GPU加速训练，也就是在模型，x_train后面加上cuda()
model = model.cuda()
X_train_noise = X_train_noise.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()
#  --------------------- 3、构建去噪自编码器模型 --------------------


#  --------------------- 4、模型训练 ---------------------
# 使用dataloader载入数据，小批量进行迭代
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(X_train_noise, X_train_noise)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)


loss_list = []
for epoch in range(5):
    running_loss = 0
    for step, (X_train_noise, X_train_noise) in enumerate(loader):
        train_prediction = model(X_train_noise)
        loss = loss_func(train_prediction, X_train_noise)  # 计算损失
        loss_list.append(loss)  # 使用append()方法把每一次的loss添加到loss_list中

        optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        #print(loss)
        running_loss += loss.item()
    else:
        print(f"第{epoch}代训练损失为：{running_loss/len(loader)}")
#  --------------------- 4、模型训练 ---------------------


#  ---------------------5、损失可视化 ---------------------
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
#  ---------------------5、损失可视化 ---------------------



