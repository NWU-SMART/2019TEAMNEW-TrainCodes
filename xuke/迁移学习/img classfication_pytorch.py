#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.6.23
#software：pycharm
#项目名称：图像分类（pytorch）
#--------------         开发者信息--------------------------

# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、读取数据及与图像预处理
# 3、迁移学习建模
# 4、训练
# 5、模型可视化与保存模型
# 6、训练过程可视化
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import torch
import torch.nn as nn
import torchvision.models as models
import os
import cv2
import gzip
import numpy as np
from torch.autograd import Variable
#  -------------------------- 1、导入需要包 -------------------------------

#  --------------------- 2、读取数据及与图像预处理 ---------------------

path = 'D:\\keras\\t10k-images-idx3-ubyte'  # 数据集地址
def load_data():
    paths = [path+'train-labels-idx1-ubyte.gz', path+'train-images-idx3-ubyte.gz',
             path+'t10k-labels-idx1-ubyte.gz', path+'t10k-images-idx3-ubyte.gz'
            ]
    # 解压训练标签
    with gzip.open(paths[0], 'rb') as lbpath:
         y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 解压训练图像
    with gzip.open(paths[1], 'rb') as imgpath:
         x_train = np.frombuffer(
         imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    # 解压测试标签
    with gzip.open(paths[2], 'rb') as lbpath:
         y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    # 解压测试图像
    with gzip.open(paths[3], 'rb') as imgpath:
         x_test = np.frombuffer(
         imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

# 调用函数
(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 32       # 批大小32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'keras_fashion_transfer_learning_trained_model.h5'

# 由于mist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,因为扩充一下mnist的最后一维
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
x_train = np.asarray(x_train)  # 变为数组
x_test = np.asarray(x_test)
x_train = x_train.astype('float32')  # 变为浮点型
x_test = x_test.astype('float32')
x_train /= 255  # 数据归一化
x_test /= 255   # 数据归一化
x_train = Variable(torch.from_numpy(x_train))  # x_train变为variable数据类型
x_test = Variable(torch.from_numpy(x_test))    # x_test变为variable数据类型
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#  --------------------- 2、读取数据及与图像预处理 ---------------------

#  -------------------------- 3、迁移学习建模 -------------------------

class VGGNet(nn.Module):
    def __init__(self,num_classes=10):
        super(VGGNet,self).__init__()
        net = models.vgg16(pretrained=True)     # 从预训练模型加载VGG16网络参数
        net.classifier = nn.Sequential()        # 将分类层置空，下面加进我们的分类层
        for parma in net.parameters():
            parma.requires_grad = False         # 不计算梯度，不会进行梯度更新
        self.feature = net                      # 保留vgg16的特征层
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,256),             # 512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,num_classes),         # 分为10类
            nn.Softmax()
        )

    def forward(self,x):
        x = x.permute(0,3,1,2) # 例如  input[60000, 28, 28, 1]---->[6000,1,28,28] 交换位置
        x = self.feature(x)
        x = x.view(x.size(0),-1)  # 拉平，作用相当于Flatten
        x = self.classifier(x)
        return x
model = VGGNet()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

#  -------------------------- 3、迁移学习建模 -------------------------
#  -------------------------- 4、模型训练 -------------------------

for t in range(5):
    x = model(x_train)          # 向前传播
    loss = loss_fn(x, y_train)  # 计算损失
    if (t + 1) % 1 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(t + 1, 5, loss.item()))  # 每训练1个epoch，打印一次损失函数的值
    optimizer.zero_grad()      # 在进行梯度更新之前，先使用optimier对象提供的清除已经积累的梯度
    loss.backward()            # 计算梯度
    optimizer.step()           # 更新梯度
    if (t + 1) % 5 == 0:
        torch.save(model.state_dict(), "./pytorch_imageclassification_transferlearning_model.h5")  # 每5个epoch保存一次模型
        print("save model")

#  -------------------------- 4、模型训练 -------------------------