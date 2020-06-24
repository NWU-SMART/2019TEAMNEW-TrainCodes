
import torch
import numpy as np
from matplotlib import pyplot as plt

def one_hot(label, depth=10): #label转onehot （独热码:有多少个状态就有多少位置，每个位置是出现的概率，第一个位置一般表示0
    # 故1., 0., 0., ..., 0., 0., 0.表示是0的概率为1）
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out

# step1. load dataset

x = np.load(r'npy_data/X.npy')
y = np.load(r'npy_data/Y.npy')

#array转变为tensor
x = torch.tensor(x)
y = torch.tensor(y)

#squeeze(n):去除第n维，只能去除数值为1的dim
#unsqueeze(n):增加第n维，数值默认为1
x = x.unsqueeze(1)

#将one_hot转化为label
y = torch.topk(y, 1)[1].squeeze(1)

#修改标签
dict = torch.tensor([9,0,7,6,1,8,4,3,2,5])
y = dict[y]

#print(x.shape, y.shape, x.min(), x.max())
#torch.Size([2062, 1, 64, 64]) torch.Size([2062]) tensor(0.0039) tensor(1.)

#产生随机打散种子
idx = torch.randperm(2062)
#print(idx,idx.shape)

#x、y协同打散
x = x[idx]
y = y[idx]

#将label转化为one_hot
y = one_hot(y)

#数据切片
x_1 = x[:500]
x_2 = x[500:1000]
y_1 = y[:500]
y_2 = y[500:1000]

#制造重叠手势
x = x_1 + x_2
y = y_1 + y_2 #标签变为two-hot

#print(y.shape) #torch.Size([500, 10])
#形成双标签
y = torch.topk(y, 2)[1] #torch.topk(y, k, dim)，似乎默认dim=1；由于这里取的是top2，所以也没必要使用squeeze
#print(y.shape) #torch.Size([500, 2])

print("生成图片...")
plt.imshow(x[5][0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} and {}".format("mixed", y[5][0].item(), y[5][1].item()))
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来

#end of file

