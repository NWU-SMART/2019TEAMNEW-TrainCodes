#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.25
#software：pycharm
#项目名称：图像回归（keras）
#--------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取手写体数据及与图像预处理
# 3、构建自编码器模型
# 4、模型可视化
# 5、训练
# 6、查看自编码器的压缩效果
# 7、查看自编码器的解码效果
# 8、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------

import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms, models
import gzip
import numpy as np
import cv2
#  -------------------------- 1、导入需要包 -------------------------------
#  --------------------- 2、读取手写体数据及与图像预处理 --------------------

path = 'D:\keras\迁移学习\mnist.npz'
f = np.load(path)          # 打开文件
x_train = f['x_train']     # 训练数据
y_train = f['y_train']     # 训练数据标签
x_test = f['x_test']       # 测试数据
y_test = f['y_test']       # 测试数据标签
f.close()
# 归一化处理
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# np.prod是将28*28矩阵转化成1*784向量
x_train = x_train.reshape(len(x_train), np.prod(x_train.shape[1:]))
x_test = x_test.reshape(len(x_test), np.prod(x_test.shape[1:]))

# 伪造回归数据，转成DataFrame格式方便数据处理
y_train_pd = pd.DataFrame(y_train)
y_test_pd = pd.DataFrame(y_test)
# 设置列名
y_train_pd.columns = ['label']
y_test_pd.columns = ['label']

mean_value_list = [45, 57, 85, 99, 125, 27, 180, 152, 33]  # 均值列表
def setting_clothes_price(row):
    price = sorted(np.random.normal(mean_value_list[int(row)], 3, size=1))[0]  # 均值mean，标准差std，数量
    return np.round(price, 2)
y_train_pd['price'] = y_train_pd['label'].apply(setting_clothes_price)
y_test_pd['price'] = y_test_pd['label'].apply(setting_clothes_price)

min_max_scaler = MinMaxScaler()    # 训练集归一化
min_max_scaler.fit(y_train_pd)
y_train_label = min_max_scaler.transform(y_train_pd)[:, 1]

min_max_scaler.fit(y_test_pd)      # 验证集归一化
y_test = min_max_scaler.transform(y_test_pd)[:, 1]
y_test_label = min_max_scaler.transform(y_test_pd)[:, 0]

y_train = Variable(torch.from_numpy(y_train_label))    # 变为variable数据类型
y_test = Variable(torch.from_numpy(y_test_label))
#  --------------------- 2、读取手写体数据及与图像预处理 ---------------------

#  ----------------------------- 3、迁移学习建模 ---------------------------

class VGG(nn.Module):
    def __init__(self,num_classes=1):
        super(VGGNet,self).__init__()
        net = models.vgg16(pretrained=True)          # 加载VGG16网络参数
        net.classifier = nn.Sequential()             # 将分类层置空，下面加进我们的分类层
        for parma in net.parameters():
            parma.requires_grad = False              # 不计算梯度，不会进行梯度更新
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,256),                  # 512 * 7 * 7不能改变 ，由VGG16网络决定的
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,1),
            nn.ReLU()
        )
    def forward(self,x):
        x = x.permute(0,3,1,2)    # 例如  input[60000, 28, 28, 1]---->[6000,1,28,28] 交换位置
        x = self.feature(x)
        x = x.view(x.size(0),-1)  # 拉平=Flatten
        x = self.classifier(x)
        return x

model = VGG()
print(model)
#  ----------------------------- 3、迁移学习建模 ---------------------------

#  ---------------------- 4、模型训练---------------------------------------
loss_function = torch.nn.CrossEntropyLoss()                              # 定义损失函数
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)  # 优化器
for epoch in range(5):
    output = model(x_train)                # 输入训练数据，输出结果
    loss = loss_function(output, y_train)  # 输出和训练数据计算损失函数
    optimizer.zero_grad()                  # 梯度清零
    loss.backward()                        # 反向传播
    optimizer.step()                       # 梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 5, loss.item()))  # 每训练1个epoch，打印一次损失函数的值
#  ---------------------- 4、模型训练---------------------------------------