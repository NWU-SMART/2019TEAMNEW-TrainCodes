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
from utils import plot_image, plot_curve, one_hot, label as to_label
import random

NUM_CLASSES = 10
NUM_EPOCHS = 1
NUM_ROUTING_ITERATIONS = 3
TRAIN_BATCH_SIZE = 50
TEST_BATCH_SIZE = 50

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
        if self.num_route_nodes != -1: #routing(digitcaps)
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
        else: #no routing(primcaps)
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

        # [b, 160] => [b, 4096]
        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 4096),
            nn.Sigmoid()
        )

    def forward(self, x, y=None, y_1=None, y_2=None):
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
            indices1 = torch.kthvalue(classes, 10, dim=1)[1]
            indices2 = torch.kthvalue(classes, 9, dim=1)[1]
            y_1 = one_hot(indices1)
            y_2 = one_hot(indices2)
            # 计算后的y_1、y_2为one-hot形式，[b, 10]
        
        # y:[b, 10] => [b, 10, 1]
        # x:[b, 10, 16] * y:[b, 10, 1] => [b, 10, 16] 注意:这里是普通乘法运算，将dim=2上的1个数分别与16个数相乘，得到16个数形成新的dim=2
        # flatten:[b, 10, 16] => [b, 160]
        # 问题：reconstructions为什么要用x*y？答：此操作相当于筛选x的10个点里面与标签one-hot的值为1.的维度相同的那一个16D向量！
        # reconstruction:[b, 160] => [b, 784]
        reconstructions1 = self.decoder((x * y_1[:, :, None]).contiguous().view(x.size(0), -1))
        reconstructions2 = self.decoder((x * y_2[:, :, None]).contiguous().view(x.size(0), -1))

        return classes, reconstructions1, reconstructions2, y_1, y_2


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions1, reconstructions2):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions1)
        images = images.view(reconstructions1.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions1, images) + self.reconstruction_loss(reconstructions2, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)



#-------------- step1. load dataset --------------

device = torch.device('cuda')

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
y = to_label(y)

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

x_train = x[:1000]
x_test = x[1000:2000]
y_train = y[:1000]
y_test = y[1000:2000]
"""

#数据切片 #1、2为训练集，3、4为测试集
x_1 = x[:500]
x_2 = x[500:1000]
x_3 = x[1000:1500]
x_4 = x[1500:2000]
y_1 = y[:500]
y_2 = y[500:1000]
y_3 = y[1000:1500]
y_4 = y[1500:2000]

#全部数据搬到GPU上
x_1, x_2, x_3, x_4 = x_1.to(device), x_2.to(device), x_3.to(device), x_4.to(device)
y_1, y_2, y_3, y_4 = y_1.to(device), y_2.to(device), y_3.to(device), y_4.to(device)

#制造重叠手势及标签
#注意：重构计算用one-hot，精确度计算用label
x_train = x_1 + x_2 
x_test = x_3 + x_4
#这三个one-hot训练用
y_1_one_hot = one_hot(y_1)
y_2_one_hot = one_hot(y_2)
y_train_two_hot = y_1_one_hot + y_2_one_hot


#-------------- step2. create net --------------

net = CapsuleNet().to(device)

capsule_loss = CapsuleLoss().to(device)

optimizer = Adam(net.parameters(), lr=0.01)



#-------------- step3. training --------------

net.train()

train_loss = [] #train_loss为list类型；train_loss为画曲线图用


"""
#print("生成训练图片...")
#plot_image(x, y, 'train image', 1799)
print("生成训练图片...")
plt.imshow(x[5][0], cmap='winter', interpolation='none') #imshow()函数实现热图绘制
plt.title("{}: {} ".format("train image", y[5].item()))
plt.xticks([]) #x轴坐标设置为空
plt.yticks([]) #y轴坐标设置为空
plt.show() #将plt.imshow()处理后的图像显示出来
"""

print("training...")

batch_size = TRAIN_BATCH_SIZE 

for epoch in range(NUM_EPOCHS): #开始训练，range括号内为对数据集迭代的次数

    for batch_idx in range(10):

        x_batch = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
        y_batch = y_train_two_hot[batch_idx*batch_size:(batch_idx+1)*batch_size]
        y_1_batch = y_1_one_hot[batch_idx*batch_size:(batch_idx+1)*batch_size]
        y_2_batch = y_2_one_hot[batch_idx*batch_size:(batch_idx+1)*batch_size]
        

        classes, reconstructions1, reconstructions2, _, _ = net(x_batch, y_batch, y_1_batch, y_2_batch) #正向传播
        loss = capsule_loss(x_batch, y_batch, classes, reconstructions1, reconstructions2) #计算loss函数
        #print(loss.shape) #torch.Size([])
        
        # 更新三连
        optimizer.zero_grad() #清零上个batch的梯度信息
        loss.backward() #反向传播，计算梯度
        optimizer.step() #更新参数

        train_loss.append(loss.item()) #只有shape为[1]的tensor才可以用item函数转化成scalar
        print("epoch:", epoch, "batch:", batch_idx, "loss:", loss.item()) #输出计算过程

"""
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
"""


#-------------- step4. testing --------------

net.eval()

#---- 训练集测试 & 计算训练集精度 ----

print("testing train_data...")

batch_size = TEST_BATCH_SIZE 

total_correct = 0
total_num = 500

for batch_idx in range(10):

    x_batch = x_train[batch_idx*batch_size:(batch_idx+1)*batch_size]
    y_1_batch = y_1[batch_idx*batch_size:(batch_idx+1)*batch_size]
    y_2_batch = y_2[batch_idx*batch_size:(batch_idx+1)*batch_size]

    #pred1, pred2:为了计算acc特意留的返回值
    classes, reconstructions1, reconstructions2, pred1, pred2 = net(x_batch)
    pred1, pred2 = to_label(pred1), to_label(pred2)
    #print(pred1.shape) #torch.Size([50])

    #stack创建一个新维度再合并
    pred = torch.stack([pred1, pred2], dim=1)
    label = torch.stack([y_1_batch, y_2_batch], dim=1)

    #排序 #return : sorted, indices 
    pred, _ = torch.sort(pred, dim=1)
    label, _ = torch.sort(label, dim=1)

    #train集准确率的计算：(最大+次大)/2
    total_correct += pred.eq(label).sum().float().item() * 0.5
    
    #重构数据转化成图片格式 [b, 4096] => [b, 1, 64, 64] 

    rec1 = torch.rand(batch_size, 64, 64)
    for i in range (batch_size):
      for j in range (64):
        for k in range (64):
          rec1[i][j][k] = reconstructions1[i][j*64+k]
    rec1 = rec1.unsqueeze(1)

    rec2 = torch.rand(batch_size, 64, 64)
    for i in range (batch_size):
      for j in range (64):
        for k in range (64):
          rec2[i][j][k] = reconstructions2[i][j*64+k]
    rec2 = rec2.unsqueeze(1)

    #转化成array
    rec1 = np.array(rec1.cpu().detach().numpy())
    rec2 = np.array(rec2.cpu().detach().numpy())
    
    #保存数据为npy格式

    a = chr(batch_idx+48)
    print("saving reconstructions_data_train", a, "...")
    
    name1 = "./overlap/train_rec1_" + a + ".npy"
    name2 = "./overlap/train_rec2_" + a + ".npy"
    np.save(name1, rec1)
    np.save(name2, rec2)
    
    print("done")


acc = total_correct / total_num
print('train acc:', acc)

#---- 测试集测试 & 计算测试集精度 ----

print("testing test_data...")

batch_size = TEST_BATCH_SIZE

total_correct = 0
total_num = 500

for batch_idx in range(10):

    x_batch = x_test[batch_idx*batch_size:(batch_idx+1)*batch_size]
    y_3_batch = y_3[batch_idx*batch_size:(batch_idx+1)*batch_size]
    y_4_batch = y_4[batch_idx*batch_size:(batch_idx+1)*batch_size]
    
    #pred1, pred2:为了计算acc特意留的返回值
    classes, reconstructions1, reconstructions2, pred1, pred2 = net(x_batch)
    pred1, pred2 = to_label(pred1), to_label(pred2)
    #print(pred1.shape) #torch.Size([50])

    #stack创建一个新维度再合并
    pred = torch.stack([pred1, pred2], dim=1)
    label = torch.stack([y_3_batch, y_4_batch], dim=1)

    #排序 #return : sorted, indices 
    pred, _ = torch.sort(pred, dim=1)
    label, _ = torch.sort(label, dim=1)

    #train集准确率的计算：(最大+次大)/2
    total_correct += pred.eq(label).sum().float().item() * 0.5
    print("batch:", batch_idx, "total_correct:", total_correct)

    #重构数据转化成图片格式 [b, 4096] => [b, 1, 64, 64] 

    rec1 = torch.rand(batch_size, 64, 64)
    for i in range (batch_size):
      for j in range (64):
        for k in range (64):
          rec1[i][j][k] = reconstructions1[i][j*64+k]
    rec1 = rec1.unsqueeze(1)

    rec2 = torch.rand(batch_size, 64, 64)
    for i in range (batch_size):
      for j in range (64):
        for k in range (64):
          rec2[i][j][k] = reconstructions2[i][j*64+k]
    rec2 = rec2.unsqueeze(1)

    #转化成array
    rec1 = np.array(rec1.cpu().detach().numpy())
    rec2 = np.array(rec2.cpu().detach().numpy())
    
    #保存数据为npy格式
    
    a = chr(batch_idx+48)
    print("saving reconstructions_data_test", a, "...")
    
    name1 = "./overlap/test_rec1_" + a + ".npy"
    name2 = "./overlap/test_rec2_" + a + ".npy"
    np.save(name1, rec1)
    np.save(name2, rec2)
    
    print("done")


acc = total_correct / total_num
print('test acc:', acc)


#转化成array 
x_train = np.array(x_train.cpu().detach().numpy())
y_1 = np.array(y_1.cpu().detach().numpy())
y_2 = np.array(y_2.cpu().detach().numpy())   
x_test = np.array(x_test.cpu().detach().numpy())
y_3 = np.array(y_3.cpu().detach().numpy())
y_4 = np.array(y_4.cpu().detach().numpy())

#保存数据为npy格式
print("saving mixed_data...")
np.save('./overlap/x_train.npy',x_train)
np.save('./overlap/y_1.npy',y_1)
np.save('./overlap/y_2.npy',y_2)
np.save('./overlap/x_test.npy',x_test)
np.save('./overlap/y_3.npy',y_3)
np.save('./overlap/y_4.npy',y_4)
print("done")



#end of file


