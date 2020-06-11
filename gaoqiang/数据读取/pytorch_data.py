# ----------------------------------------------开发者信息-------------------------------------------------------------#
# 开发者：高强
# 开发日期：2020.06.10
# 开发框架：pytorch
# 适用场景：大数据量，使用 shuffle, 分割成mini-batch 等操作的时候
#----------------------------------------------------------------------------------------------------------------------#
'''
# 代码功能：
在pytorch中进行数据读取的一般有三个类：Dataset，DataLoader，DataLoaderIter
1.Dataset位于torch.utils.data.Dataset，每当我们自定义类MyDataset必须要继承它并实现其两个成员函数：
        __len__()
        __getitem__()
2.DataLoader位于torch.utils.data.DataLoader, 为我们提供了对Dataset的读取操作
    torch.nn.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dataset : 上面所实现的自定义类Dataset
    batch_size : 默认为1，每次读取的batch的大小
    shuffle : 默认为False， 是否对数据进行shuffle操作(简单理解成将数据集打乱)
    num_works : 默认为0，表示在加载数据的时候每次使用子进程的数量，即简单的多线程预读数据的方法

达到分批处理数据的目的：
Dataloder的目的是将给定的n个数据, 经过Dataloader操作后, 在每一次调用时调用一个小batch, 如:给出的是: (5000,28,28) 表示有
5000个样本,每个样本的size为(28,28),经过Dataloader处理后, 一次得到的是(100,28,28) ((假设batch_size大小为100), 表示本次取
出100个样本, 每个样本的size为(28,28)

'''

from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
from torch.autograd import Variable

class MyDataset(Dataset):
    def __init__(self):                                                   # 初始化
        xy = np.loadtxt('diabetes.csv',delimiter=',',dtype=np.float32) # 使用numpy读取数据
        self.x_data = torch.from_numpy(xy[:,0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
        self.len = xy.shape[0]

    def __len__(self):                                                    # 返回数据的长度
        return self.len

    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]

ds = MyDataset()                                                          # 通过实例化对象来访问该类

dl = DataLoader(dataset=ds,batch_size=50,shuffle=True,num_workers=0)     # 传给DataLoader

for epoch in range(2):
    for i,data in enumerate(dl):
        inputs,labels = data                                           # 将数据从dl中读出来,一次读取的样本数是32个
        inputs,labels = Variable(inputs),Variable(labels)              # 将这些数据转换成Variable类型
        # 接下来就是跑模型的环节了，这里使用print来代替
        print("epoch：", epoch, "的第", i, "个inputs", inputs.data.size(), "labels", labels.data.size())

# 共759个数据



