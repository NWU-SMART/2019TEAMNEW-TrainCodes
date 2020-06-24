"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""
import sys
sys.setrecursionlimit(15000) #手工设置递归调用深度

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from torch.autograd import Variable

from torch.optim import Adam
from matplotlib import pyplot as plt 
from utils import plot_image, plot_curve, one_hot, label as to_lable
import random

NUM_CLASSES = 10
NUM_EPOCHS = 250
NUM_ROUTING_ITERATIONS = 3
TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size() # 单下划线：“不关心的”变量

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1: #routing
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)) #定义w_ij，并将它设置为待更新参数
        else: #no routing
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)]) #定义conv层（num_capsules=8） #for _ in range 无循环数的循环表示方法

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1: #routing
            # x: [b, 32*6*6, 8] => [1, b, 32*6*6, 1, 8]
            # w: [10, 32*6*6, 8, 16] => [10, 1, 32*6*6, 8, 16]
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :] #u_hat = u @ w_ij

            logits = Variable(torch.zeros(*priors.size())).cuda() #b_ij
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2) #c_ij = softmax(b_ij) #dim=2为32*6*6,代表初级胶囊个数
                # [10, b, 32*6*6, 1, 16] => [10, b, 1, 1, 16]
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True)) #s_j = sum(c_ij * u_hat)(dim=32*6*6), v_j = squash(s_j)

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits #b_ij = b_ij + u_hat * v_j
        else: #no routing
            # x: [b, 256, 20, 20] => [b, 32*6*6, 8]
            # 第一句：创建一个list，list中有8个元素，每个元素都是一个tensor，该tensor将x打平后再在最后加上了一个维度
            # capsule(x): [b, 256, 20, 20] => [b, 32, 6, 6]
            # view: [b, 32, 6, 6] => [b, 32*6*6, 1]
            # 第二句：将list中元素concat（在现有维度上合并），在上一句增加的那个维度上合并
            # cat: [b, 32*6*6, 1] => [b, 32*6*6, 8]
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules] #self.capsules=8, 走8遍conv层后再重构
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()

        # [b, 1, 64, 64] => [b, 1, 28, 28]
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=2)
        # [b, 1, 28, 28] => [b, 256, 20, 20]
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=9, stride=1)
        # [b, 256, 20, 20] => [b, 32*6*6, 8]
        # num_route_nodes=-1时，num_capsules：输出胶囊数，其余为卷积层参数
        # num_route_nodes!=-1时，num_route_nodes：输入胶囊数，num_capsules：输出胶囊数，in_channels：输入胶囊大小，out_channels：输出胶囊大小
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        # [b, 32*6*6, 8] => [64, b, 1, 1, 16]
        self.digit_capsules = CapsuleLayer(num_capsules=64, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)
        # [b, 64, 16] => [10, b, 1, 1, 16]
        self.digit_capsules2 = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=64, in_channels=16,
                                           out_channels=16)

        # [b, 160] => [b, 4096] #[b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv0(x), inplace=True)
        x = F.relu(self.conv1(x), inplace=True) #inplace:选择是否进行覆盖运算
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1) #[64, b, 1, 1, 16] => [b, 64, 16]
        x = self.digit_capsules2(x).squeeze().transpose(0, 1) #[10, b, 1, 1, 16] => [b, 10, 16]

        classes = (x ** 2).sum(dim=-1) ** 0.5 #在16这个维度上，取L2范数，即向量的长度 [b, 10, 16] => [b, 10]
        classes = F.softmax(classes, dim=-1) #在10这个维度上，转化为概率值 [b, 10]

        # train调用的是net(x, y) , test调用的是net(x)
        if y is None: # when testing, not training  #将概率分类classes转化为预测值y
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data) #eye为单位矩阵
            # 计算后的y为one - hot形式，[b, 10]

        # y:[b, 10] => [b, 10, 1]
        # [b, 10, 16] * [b, 10, 1] => [b, 10, 16] 注意:这里是普通乘法运算，将dim=2上的1个数分别与16个数相乘，得到16个数形成新的dim=2
        # flatten:[b, 10, 16] => [b, 160]
        # 问题：reconstructions为什么要用x*y？答：此操作相当于筛选x的10个点里面与标签one-hot的值为1.的维度相同的那一个16D向量！
        # reconstruction:[b, 160] => [b, 784]
        """
        print(x.shape, y.shape) # torch.Size([60, 10, 16]) torch.Size([60, 10])
        z = x * y[:, :, None]
        print(z.shape) # torch.Size([60, 10, 16])
        #RuntimeError:view size is not compatible with input tensor's size and stride
        #原因：用多卡训练的时候tensor不连续，即tensor分布在不同的内存或显存中。
        #解决方法：对tensor进行操作时先调用contiguous()。如tensor.contiguous().view()。
        z = z.contiguous().view(z.size(0), -1) 
        reconstructions = self.decoder(z)
        """
        reconstructions = self.decoder((x * y[:, :, None]).contiguous().view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)



# step1. load dataset

x = np.load(r'npy_data/X.npy')
y = np.load(r'npy_data/Y.npy')

#array转变为tensor
x = torch.tensor(x)
y = torch.tensor(y)

#squeeze(n):去除第n维，只能去除数值为1的dim
#unsqueeze(n):增加第n维，数值默认为1
x = x.unsqueeze(1)

#数据增强
x = augmentation(x)

#将one_hot转化为label
y = to_lable(y)

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
"""
#数据切片
x_train = x[:1800] #从第0张到第1799张
x_test = x[1800:] #从第1800张到最后一张
y_train = y[:1800]
y_test = y[1800:]
"""
x_train = x[:1000]
x_test = x[1000:2000]
y_train = y[:1000]
y_test = y[1000:2000]

# step2. create net

device = torch.device('cuda')

net = CapsuleNet().to(device)

capsule_loss = CapsuleLoss().to(device)

optimizer = Adam(net.parameters(), lr=0.01)

# step3. training

net.train()

train_loss = [] #train_loss为list类型；train_loss为画曲线图用

x = x_train
y = y_train

#print("生成训练图片...")
#plot_image(x, y, 'train image', 1799)
print("生成训练图片...")
plt.imshow(x[5][0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} ".format("train image", y[5].item()))
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来

x, y = x.to(device), y.to(device)

batch_size = TRAIN_BATCH_SIZE #100

for epoch in range(NUM_EPOCHS): #开始训练，range括号内为对数据集迭代的次数

    for batch_idx in range(10):

        x_batch = x[batch_idx*batch_size:(batch_idx+1)*batch_size]
        y_batch = y[batch_idx*batch_size:(batch_idx+1)*batch_size]
        y_onehot = one_hot(y_batch)

        classes, reconstructions = net(x_batch, y_onehot) #正向传播
        loss = capsule_loss(x_batch, y_onehot, classes, reconstructions) #计算误差(代价函数)
        #print(loss.shape) #torch.Size([])
        
        optimizer.zero_grad() #清零梯度
        loss.backward() #反向传播，计算梯度
        optimizer.step() #更新参数

        train_loss.append(loss.item()) #只有shape为[1]的tensor才可以用item函数转化成scalar
        print("epoch:", epoch, "batch:", batch_idx, "loss:", loss.item()) #输出计算过程

#train_loss = train_loss.cpu()
#print("生成loss随epoch变化曲线图...")
#plot_curve(train_loss)
print("生成loss随epoch变化曲线图...")
fig = plt.figure()
plt.plot(range(len(train_loss)), train_loss, color='blue')
plt.legend(['value'], loc='upper right')
plt.xlabel('step')
plt.ylabel('value')
plt.show()

# step4. testing

net.eval()

#训练集测试 & 计算训练集精度

x, y = x_train, y_train

x, y = x.to(device), y.to(device)

batch_size = TEST_BATCH_SIZE #100

total_correct = 0
total_num = 1000

for batch_idx in range(10):

    x_batch = x[batch_idx*batch_size:(batch_idx+1)*batch_size]
    y_batch = y[batch_idx*batch_size:(batch_idx+1)*batch_size]

    classes, reconstructions = net(x_batch)
    pred = classes.argmax(dim=1) # argmax:返回最大数的索引

    total_correct += pred.eq(y_batch).sum().float().item()

acc = total_correct / total_num
print('train acc:', acc)

#测试集测试 & 计算测试集精度 

x, y = x_test, y_test

x, y = x.to(device), y.to(device)

batch_size = TEST_BATCH_SIZE #100

total_correct = 0
total_num = 1000

for batch_idx in range(10):

    x_batch = x[batch_idx*batch_size:(batch_idx+1)*batch_size]
    y_batch = y[batch_idx*batch_size:(batch_idx+1)*batch_size]

    classes, reconstructions = net(x_batch)
    pred = classes.argmax(dim=1) # argmax:返回最大数的索引

    total_correct += pred.eq(y_batch).sum().float().item()

acc = total_correct / total_num
print('test acc:', acc)

x_batch, pred = x_batch.cpu(), pred.cpu()
#print("生成测试图片...")
#plot_image(x, pred, 'test image', 261)
print("生成测试图片...")
plt.imshow(x_batch[5][0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} ".format("test image", pred[5].item()))
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来

# 保存reconstructions：

# [100, 4096] => [100, 1, 64, 64] 转化成图片格式
rec = torch.rand(100, 64, 64)
for i in range (100):
  for j in range (64):
    for k in range (64):
      rec[i][j][k] = reconstructions[i][j*64+k]
rec = rec.unsqueeze(1)

#转化成array    
rec = np.array(rec.cpu().detach().numpy())

#保存数据为npy格式
np.save('./rec_data_test/rec.npy',rec)


#end of file


