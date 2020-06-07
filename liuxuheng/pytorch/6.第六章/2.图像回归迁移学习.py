# ----------------开发者信息----------------------------
# 开发者：刘盱衡
# 开发日期：2020年5月28日
# 修改日期：
# 修改人：
# 修改内容：
# ----------------开发者信息----------------------------

#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
import torch
import cv2
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------
path = 'D:\\keras_datasets\\mnist.npz'#数据地址
f = np.load(path)#读入数据
# 训练数据
X_train=f['x_train']
y_train=f['y_train']
# 测试数据
X_test=f['x_test']
y_test=f['y_test']
f.close()
# 给数据增扩充通道维度
X_train = [cv2.cvtColor(cv2.resize(i, (28, 28)), cv2.COLOR_GRAY2RGB) for i in X_train] 
X_test = [cv2.cvtColor(cv2.resize(i, (28, 28)), cv2.COLOR_GRAY2RGB) for i in X_test]
# 将数据变为array数组类型
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
# 变为variable数据类型
x_train = Variable(torch.from_numpy(X_train))
x_test = Variable(torch.from_numpy(X_test))
# 变为channel_first
x_train = x_train.permute(0,3,2,1)
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
#  --------------------- 3、伪造回归数据 ---------------------

#  --------------------- 4、数据归一化 ---------------------
# 训练集归一化
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)[:, 1]
# 验证集归一化
min_max_scaler.fit(y_test_pd)
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]  # 归一化后的标签
y_train = Variable(torch.from_numpy(y_train))
#  --------------------- 4、数据归一化 ---------------------

#  --------------------- 5、迁移学习建模 ---------------------
model = models.vgg16(pretrained=True)# 下载VGG16，并通过设置参数pretrained=True，下载模型附带了已经优化好的模型参数。
for parma in model.parameters():
    parma.requires_grad = False # 将参数parma.requires_grad全部设置为False，这样对应的参数将不计算梯度，不会进行梯度更新
# 定义新的全连接层结构并赋值给model.classifier，在完成新的全连接层定义之后，全连接层中的parma.requires_grad参数会被默认重置为True
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 256), # 7x7x512 -----> 256
                                      torch.nn.ReLU(),
                                      torch.nn.Dropout(p=0.5),
                                      torch.nn.Linear(256, 1), # 256 ----> 1
                                      torch.nn.ReLU())
print(model)# 查看模型
#  --------------------- 5、迁移学习建模 ---------------------

#  -------------------------- 6、训练模型   --------------------------------
loss_function = torch.nn.CrossEntropyLoss() # 损失函数
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001) # 优化器

for epoch in range(5):
    output = model(x_train)# 输入训练数据，获取输出
    loss = loss_function(output, y_train)# 输出和训练数据计算损失函数
    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播
    optimizer.step()#梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 5, loss.item()))#每训练1个epoch，打印一次损失函数的值
#  -------------------------- 6、训练模型   --------------------------------

