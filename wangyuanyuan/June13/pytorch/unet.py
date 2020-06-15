#---------------------------------------------------------开发者信息-----------------------------------------------------
#开发者：王园园
#开发日期：2020.6.12
#开发软件：pycharm
#开发项目：Unet实现医学图像分割（pytorch）

#------------------------------------------------------------导包-------------------------------------------------------
import argparse
import os
import torch
from PIL.Image import Image
from networkx.drawing.tests.test_pylab import plt
from torch import nn, optim
from torch.testing._internal.common_utils import args
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

#--------------------------------------------------------------处理数据--------------------------------------------------

def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2                     #os.listdir(path)返回指定路径下的文件和文件夹列表
    for i in range(n):
        img = os.path.join(root, '%%03d.png' % i)
        mask = os.path.join(root, '%03d_mask.png' % i)
        imgs.append([img, mask])                       #append只能有一个参数，加上[]变成一个list
    return imgs

#data.Dataset:
#所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)
class LiverDataset(data.Dataset):
    # 创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y     #返回的是图片

    def __le__(self):
        return len(self.imgs)

#-----------------------------------------------------------构建模型-----------------------------------------------------
#把常用的2个卷积操作简单封装
class DoubleConv(nn.Module):
    def __int__(self, in_ch, out_ch):
        super(DoubleConv, self).__int__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),             #in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        #逆卷积
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)          #按维数1（列）拼接,列增加
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out        #化成(0~1)区间

#---------------------------------------------------------训练模型-------------------------------------------------------
# 是否使用current cuda device or torch.device('cuda:0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#把多个步骤整合到一起， channel=(channel-mean)/std, 分别对三个通道处理
x_transforms = transforms.Compose([
    transforms.ToTensor(),          # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])       #标准化至[-1,1],规定均值和标准差
])

#mask只需要转换为tensor
y_transforms = transforms.ToTensor()

#参数解析器，用来解析从终端读取的命令
parse = argparse.ArgumentParser()

def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0                   #minibatch数
        for x, y in dataload:      #分100次遍历数据集，每次遍历batch_size=4
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()  #每次minibatch都要将梯度(dw,db,...)清零
            outputs = model(inputs) #前向传播
            loss = criterion(outputs, labels)  #计算损失
            loss.backward()         #梯度下降，计算出梯度
            optimizer.step()        #更新参数一次：所有的优化器Optimizer都实现了step()方法来对所有的参数进行更新
            epoch_loss += loss.item()
            print('%d%d, train_loss:%0.3f' % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print('epoch %d loss:%0.3f' % (epoch, epoch_loss))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model

#训练模型
def train():
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset('data/train', transform=x_transforms, target_transform=y_transforms)
    # DataLoader:该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor
    # num_workers：表示通过多个进程来导入数据，可以加快数据导入速度
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)

#显示模型的输出结果
def test():
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    liver_dataset = LiverDataset('data/val', transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    plt.ion()
    with torch.no_grad():
        for x, _ in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()
            plt.imshow(img_y)
            plt.pause(0.1)
        plt.show()

#参数解析
parse = argparse.ArgumentParser()      #创建一个ArgumentParser对象
#添加参数
parse.add_argument('action', type=str, help='train or test')
parse.add_argument('--batch_size', type=int, default=1)
parse.add_argument('--ckp', type=str, help='the path of model weight file')
args = parse.parse_args()

#train()
#test()
args.ckp = 'weights_19.pth'    #如果见到一个.pth 文件，就会将文件中所记录的路径加入到 sys.path 设置中，于是 .pth 文件说指明的库也就可以被 Python 运行环境找到了。
test()



