#----------------------------------------------------------#
# 开发者：朱梦婕
# 开发日期：2020年6月23日
# 开发框架：Pytorch
# 开发内容：图像回归（迁移学习）
#----------------------------------------------------------#
'''
存在错误：size mismatch
解决方法：在全连接层前加AdaptiveAvgPool2d(output_size=(7, 7))
'''

# ----------------------   代码布局： ----------------------
# 1、导入 torch, numpy, gzip和 cv2的包
# 2、读取数据及与图像预处理
# 3、伪造回归数据
# 4、数据归一化
# 5、参数定义
# 6、迁移学习建模
# 7、模型训练
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import gzip
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torchvision.models as models
from sklearn.preprocessing import MinMaxScaler
import cv2
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用第2个GPU
#  -------------------------- 导入需要包 -------------------------------
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

# 导入mnist数据
# (X_train, _), (X_test, _) = mnist.load_data() 服务器无法访问
# 本地读取数据
# D:\\keras_datasets\\mnist.npz(本地路径)
path = 'mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
x_train = f['x_train']
y_train = f['y_train']   # 加载label
# 测试数据
x_test = f['x_test']
y_test = f['y_test']    # 加载label
f.close()
# 数据放到本地路径test

# 观察下X_train和X_test维度
print('x_train维度:',x_train.shape)  # 输出x_train维度  (60000, 28, 28)
print('x_test维度:',x_test.shape)   # 输出x_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  --------------------- 3、伪造回归数据 ---------------------

# 转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

# 给每一类衣服设置价格
mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 225, 33]  # 均值列表
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3,size=1))[0] #均值mean,标准差std,数量
    return np.round(price, 2)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

print(y_train_pd.head(5))
print('-------------------')
print(y_test_pd.head(5))

#  --------------------- 3、伪造回归数据 ---------------------

#  --------------------- 4、数据归一化 ---------------------

# 由于mist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,因为扩充一下mnist的最后一维
'''
cv2.resize：将原图放大到48*48
cv2.cvtColor(p1,p2) ：是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式
cv2.COLOR_GRAY2RGB 将灰度图片转化为格式
cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
'''
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

# y_train_price_pd = y_train_pd['price'].tolist()
# y_test_price_pd = y_test_pd['price'].tolist()
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]

# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
print(len(y_train))
print(len(y_test))

# 转换为tensor形式
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test)

#  --------------------- 4、数据归一化 ---------------------

#  ---------------------------------5、参数定义 --------------------------------

batch_size = 32
num_classes = 10
epochs = 5

#  ----------------------------------- 参数定义---------------------------------------------

# -------------------------------6、模型构建------------------------
x_train =x_train.permute(0, 3 , 2, 1)
class VGG16(nn.Module):


    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3) # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1)) # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3) # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1)) # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3) # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1)) # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3) # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3) # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1)) # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1)) # pooling 512 * 7 * 7
        self.avgepool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.fc3 = nn.Linear(256, 1)
        # softmax 1 * 1 * 1000

    def forward(self, x):

        # x.size(0)即为batch_size
        # in_size = x.size(0)

        out = self.conv1_1(x) # 222
        out = self.relu(out)
        out = self.conv1_2(out) # 222
        out = self.relu(out)
        out = self.maxpool1(out) # 112

        out = self.conv2_1(out) # 110
        out = self.relu(out)
        out = self.conv2_2(out) # 110
        out = self.relu(out)
        out = self.maxpool2(out) # 56

        out = self.conv3_1(out) # 54
        out = self.relu(out)
        out = self.conv3_2(out) # 54
        out = self.relu(out)
        out = self.conv3_3(out) # 54
        out = self.relu(out)
        out = self.maxpool3(out) # 28

        out = self.conv4_1(out) # 26
        out = self.relu(out)
        out = self.conv4_2(out) # 26
        out = self.relu(out)
        out = self.conv4_3(out) # 26
        out = self.relu(out)
        out = self.maxpool4(out) # 14

        out = self.conv5_1(out) # 12
        out = self.relu(out)
        out = self.conv5_2(out) # 12
        out = self.relu(out)
        out = self.conv5_3(out) # 12
        out = self.relu(out)
        out = self.maxpool5(out) # 7

        out = self.avgepool(out)

        # 展平
        out = out.view(out.size(0), -1) # out.size(0）是 batchsize

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.softmax(out)
        return out

model = VGG16()
print(model)  # 查看模型

# GPU
model = model.cuda()
x_train = x_train.cuda()
y_train = y_train.cuda()
# -------------------------------模型构建------------------------

#   ----------------------7、训练模型 --------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()


#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(epochs):
    running_loss = 0
    for step, (x_train, y_train) in enumerate(loader):
        train_prediction = model(x_train)
        loss = loss_func(train_prediction, y_train)  # 计算损失
        loss_list.append(loss) # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        running_loss += loss.item()
    else:
        print(f"Epoch{t}：{running_loss/len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()
#   ---------------------- 训练模型 --------------------------