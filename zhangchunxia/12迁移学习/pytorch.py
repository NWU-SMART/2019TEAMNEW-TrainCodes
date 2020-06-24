# ------------------------开发者信息-------------------------------------
# 开发者：张春霞
# 开发日期：2020年6月23日
# 内容:迁移学习-图像分类（pytorch)
# 修改日期：
# 修改人：
# 修改内容：
# ------------------------开发者信息--------------------------------------
# ----------------------   代码布局： ------------------------------------
# 1、导入 torch的包
# 2、读取数据及与图像预处理
# 3、迁移学习建模
# 4、训练
# 5、模型可视化与保存模型
# 6、训练过程可视化
# ----------------------   代码布局： ------------------------------------
#  ---------------------- 1、导入需要包 -----------------------------------
import gzip
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import datasets, transforms, models
#  ---------------------- 1、导入需要包 -----------------------------------
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
path = 'D:\\northwest\\小组视频\\keras_datasets\\'
def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]
    # 加载数据返回4个NumPy数组
    with gzip.open(paths[0], 'rb') as lbpath:  # 读压缩文件
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8) # frombuffer将data以流的形式读入转化成ndarray对象
        # 第一参数为stream,第二参数为返回值的数据类型，第三参数指定从stream的第几位开始读入
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)
(x_train, y_train), (x_test, y_test) = load_data()  # 载入数据
y_train = LabelBinarizer().fit_transform(y_train)  # 对y进行one-hot编码
y_test = LabelBinarizer().fit_transform(y_test)
# 由于mist的输入数据维度是(num, 28, 28)，vgg16 需要三维图像,因为扩充一下mnist的最后一维
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]
x_train = np.asarray(X_train)  # 变为array数组
x_test = np.asarray(X_test)  # 变为array数组
x_train = x_train.astype('float32')/255 # 变为浮点型,归一化
x_test = x_test.astype('float32')/255 # 变为浮点型。归一化
x_train = Variable(torch.from_numpy(x_train))  # 变为variable数据类型
y_train = Variable(torch.from_numpy(y_train))  # 变为variable数据类型
x_test = Variable(torch.from_numpy(x_test))  # 变为variable数据类型
y_test = Variable(torch.from_numpy(y_test))  # 变为variable数据类型
x_train = x_train.permute(0, 3, 2, 1)  # 变为channel_first
#  ---------------------- 2、读取手图像数据及与图像预处理 ------------------
#  ---------------------- 3、迁移学习建模-----------------------------------
model = models.vgg16(pretrained=True)  # 下载VGG16，并通过设置参数pretrained=True，下载模型附带了已经优化好的模型参数。
for parma in model.parameters():
    parma.requires_grad = False  # 将参数parma.requires_grad全部设置为False，这样对应的参数将不计算梯度，不会进行梯度更新
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 256),  # 7x7x512 -----> 256，定义三个全连接层
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(256, 10),  # 256 ----> 10
                                       torch.nn.Softmax())
print(model)            # 打印查看模型
#  ---------------------- 3、迁移学习建模-----------------------------------
#  ---------------------- 4、模型训练---------------------------------------
loss_function = torch.nn.CrossEntropyLoss()                  # 定义损失函数
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.00001)  # 优化器
for epoch in range(5):
    output = model(x_train)                # 输入训练数据，输出结果
    loss = loss_function(output, y_train)  # 输出和训练数据计算损失函数
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 梯度更新
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 5, loss.item()))  # 每训练1个epoch，打印一次损失函数的值
#  ---------------------- 4、模型训练---------------------------------------

