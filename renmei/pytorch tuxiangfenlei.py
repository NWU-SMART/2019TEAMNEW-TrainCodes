#---------------------------------开发者信息----------------------------------------
#姓名：任梅
#日期：2020.05.23
#---------------------------------开发者信息----------------------------------------

#--------------------------------代码布局———————————————————-
#1.导入需要包
#2.导入数据
#3.构建模型
#4.训练

#--------------------------------导入需要包————————————————————-
import gzip
import numpy as np
import torch.nn as nn
import torch
from torch.optim import adam
from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
#--------------------------------导入需要包————————————————————-

#--------------------------------导入数据———————————————————-
def load_data():
    paths=[
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\train-lables-idx1-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\train-images-idx3-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\t10k-labels-idx1-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\t10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0],'rb') as lbpath:
        y_train=np.frombuffer(lbpath.read(),np.unit8,offest=8)
    with gzip.open(paths[1],'rb') as imgpath:
        x_train=np.frombuffer(imgpath.read(),np.unit8,offest=16).rashape(len(y_train),28,28,1)
    with gzip.open(paths[2],'rb') as lbpath:
        y_test=np.frombuffer(lbpath.read(),np.unit8,offest=8)
    with gzip.open(paths[2],'rb') as lbpath:
        x_test=np.frombuffer(imgpath.read(),np.unit8,offest=16).reshape(len(y_test),28,28,1)
    return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test)=load_data()
#--------------------------------导入数据———————————————————-
batch_size=32
"""
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随即将图片水平翻转
    transforms.RandomRotation(15),  # 随即旋转图片15度
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # 将图片转成 Tensor
])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

train_set = ImgDataset(x_train, y_train,train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = ImgDataset(x_test, y_test,test_transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
"""


#---------------------------构建模型——————————————————--
class Net(nn.Module):
    def __init__(self,num_classes=10):
        super(Net,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),

            nn.Relu()

        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,32,3),
            nn.Relu(),
            nn.MaxPooling2d(2,2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),

            nn.Relu()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.Relu(),
            nn.MaxPooling2d(2, 2),#24*24
            nn.Dropout(0.25)
        )
        self.fc1=nn.Sequential(
            nn.Linear(64*24*24,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25)

        )
        self.fc2=nn.Sequential(
            nn.Linear(512,num_classes),
            nn.BatchNorm1d(num_classes),
            nn.Softmax()

        )
    def forward(self,x):

        x=self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x=self.fc1(x)
        x=self.fc2(x)
        return x
#---------------------------构建模型——————————————————--

model=Net(num_classes=10)
loss=nn.CrossEntropyLoss()
optimizer=adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
epoch=5
torch.save(model.state_dict(),"model{}.format(epoch)")
train_losses=[]
test_losses=[]
#--------------------------------------训练--------------------------------
def train(epoch):
    model.train()
    optimizer.zero_grad()
    loss=0.0
    output_train=model(x_train)
    output_test=model(x_test)
    loss_train=loss(output_train,y_train)
    loss_test=loss(output_test,y_test)
    train_losses.append(loss_train)
    test_losses.append(loss_test)
    loss_train.backward()
    optimizer.step()
    print('epoch:',epoch+1,'\t','loss:',loss_test)

for epoch in range(epoch):
    train()
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend()
plt.show()








