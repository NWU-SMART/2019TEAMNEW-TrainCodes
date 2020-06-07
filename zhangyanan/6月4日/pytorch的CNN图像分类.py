# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月4日
# 开发内容：
# 修改日期：
# 修改人：
# 修改内容：
'''在这里不能使用onehot编码，否则损失函数会报错
pytorch输入图片的尺寸必须是CxHxW，所以使用premute方法把[60000, 28, 28, 1]变为[60000, 28, 28, 1]
此程序必须在有gpu版本的pytorch上运行
使用dataloader载入数据，小批量迭代，否则这个数据集图片比较多，计算机算不过来'''

#  -------------------------- 导入需要包 -------------------------------
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import gzip
import matplotlib.pyplot as plt


def load_data():
    paths = [
        'D:\\Study\\project\\CNN图像识别\\train-labels-idx1-ubyte.gz',
        'D:\\Study\\project\\CNN图像识别\\train-images-idx3-ubyte.gz',
        'D:\\Study\\project\\CNN图像识别\\t10k-labels-idx1-ubyte.gz',
        'D:\\Study\\project\\CNN图像识别\\t10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

#  不使用one-hot编码，原因是torch.nn.CrossEntropyLoss()接受的目标值必须是类标值，而不是one-hot编码
#  不使用one-hot编码，原因是torch.nn.CrossEntropyLoss()接受的目标值必须是类标值，而不是one-hot编码
#  不使用one-hot编码，原因是torch.nn.CrossEntropyLoss()接受的目标值必须是类标值，而不是one-hot编码
#  不使用one-hot编码，原因是torch.nn.CrossEntropyLoss()接受的目标值必须是类标值，而不是one-hot编码
#  不使用one-hot编码，原因是torch.nn.CrossEntropyLoss()接受的目标值必须是类标值，而不是one-hot编码
#  不使用one-hot编码，原因是torch.nn.CrossEntropyLoss()接受的目标值必须是类标值，而不是one-hot编码
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # 数据归一化
x_test /= 255   # 数据归一化

x_train = Variable(torch.from_numpy(x_train))  # 参考学长的，要不然无法使用gpu加速
x_test = Variable(torch.from_numpy(x_test))
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#   ---------------------- 构建模型 ---------------------------
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu2 = torch.nn.ReLU()
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout1 = torch.nn.Dropout(0.25)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.conv4= torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu4 = torch.nn.ReLU()
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=3)
        self.dropout2 = torch.nn.Dropout(0.25)
        self.flatten = torch.nn.Flatten()
        self.dense1 = torch.nn.Linear(576, 512)
        self.relu5 = torch.nn.ReLU()
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dense2 = torch.nn.Linear(512, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # pytorch输入图片的尺寸必须是CxHxW，所以使用premute方法把[60000, 28, 28, 1]变为[60000, 28, 28, 1]
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxPool1(x)
        x = self.dropout1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxPool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x


model = ImageModel()  # 实例化模型

'''# 以下三行可以调用GPU加速训练，也就是在模型，x_train，y_train后面加上cuda()'''
model = model.cuda()
x_train = x_train.cuda()
y_train = y_train.cuda()


print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.CrossEntropyLoss()


#使用dataloader载入数据，小批量进行迭代，要不然计算机算不过来
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(x_train, y_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)

#   ---------------------- 训练模型 --------------------------
loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
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
        print(f"第{t}代训练损失为：{running_loss/len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()