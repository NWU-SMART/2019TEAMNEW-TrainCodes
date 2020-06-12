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
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\train-labels-idx1-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\train-images-idx3-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\t10k-labels-idx1-ubyte.gz',
        'C:\\Users\\Administrator\\Desktop\\代码\\CNN数据集\\t10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0],'rb') as lbpath:
        y_train=np.frombuffer(lbpath.read(),np.uint8,offset=8)
    with gzip.open(paths[1],'rb') as imgpath:
        x_train=np.frombuffer(imgpath.read(),np.uint8,offest=16).rashape(len(y_train),28,28,1)
    with gzip.open(paths[2],'rb') as lbpath:
        y_test=np.frombuffer(lbpath.read(),np.uint8,offest=8)
    with gzip.open(paths[2],'rb') as lbpath:
        x_test=np.frombuffer(imgpath.read(),np.uint8,offest=16).reshape(len(y_test),28,28,1)
    return (x_train,y_train),(x_test,y_test)

(x_train,y_train),(x_test,y_test)=load_data()
#--------------------------------导入数据———————————————————-


x_train=torch.Tensor(x_train).float()
y_train=torch.Tensor(y_train).float()
x_test=torch.Tensor(x_test).float()
y_test=torch.Tensor(y_test).float()
num_classes=10
#---------------------------构建模型——————————————————--
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,padding=1,),#28

            nn.Relu()

        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,32,kernel_size=3),#26
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2),#`13
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),#13

            nn.Relu()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),#11
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2),#5
            nn.Dropout(0.25)
        )
        self.fc1=nn.Sequential(
            nn.Linear(64*5*5,512),

            nn.ReLU(),
            nn.Dropout(0.25)

        )
        self.fc2=nn.Sequential(
            nn.Linear(512,num_classes),

            nn.Softmax()

        )
    def forward(self,x):

        x=self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x=self.fc1(x)
        out=self.fc2(x)
        return out
#---------------------------构建模型——————————————————--

model=Net()
loss=nn.CrossEntropyLoss()
optimizer=adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
epoch=1

train_losses=[]
test_losses=[]
#--------------------------------------训练--------------------------------
for i in range(epoch):
    model.train()

    loss=0.0
    output_train=model(x_train)
    loss_train=loss(output_train,y_train)
    train_losses.append(loss_train/len(x_train))
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    output_test = model(x_test)
    loss_test=loss(output_test,y_test)
    test_losses.append(loss_test/len(x_test))
    print("第%d次迭代,训练集损失为%f,验证集损失为为%f" % (i + 1, loss_train, loss_test))


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend()
plt.show()








