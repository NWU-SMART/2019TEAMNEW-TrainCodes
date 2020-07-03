from __future__ import print_function

import torch
import torch.nn as nn  # 神经网络工具箱
import torch.optim as optim  # 神经网络优化器
import torch.nn.functional as F  # 神经网络函数
import torch.backends.cudnn as cudnn  # cuDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
import torchvision  # 图像操作工具库
import transforms as transforms  # 对图像进行预处理
import numpy as np
import os  # 处理文件与目录
import time
import datetime
import argparse
import utils
from CK import CK
from torch.autograd import Variable
from models import *

'''添加参数
type:指定参数类别，默认是str，传入数字要定义
help：是一些提示信息
default：是默认值
'''
parser = argparse.ArgumentParser(description='PyTorch CK+ CNN Training')  # 创建解析器
parser.add_argument('--model', type=str, default='VGG19', help='CNN architecture')  # 模型
parser.add_argument('--dataset', type=str, default='CK+', help='dataset')  # 数据集
parser.add_argument('--fold', default=1, type=int, help='k fold number')  # 交叉验证
parser.add_argument('--bs', default=64, type=int, help='batch_size')  # 批处理数量
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')  # 学习率
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()  # 解析参数

use_cuda = torch.cuda.is_available()  # 这个指令的作用是看你电脑的 GPU 能否被 PyTorch 调用

# 初始化
best_Test_acc = 0  # best PrivateTest accuracy
best_Test_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 20  # 初始学习率衰减
learning_rate_decay_every = 10  # 每一次的学习率衰减
learning_rate_decay_rate = 0.99  # 学习率衰减衰减指数0.99

cut_size = 96  # 剪裁尺寸96*96
total_epoch = 2000  # 训练数据集中所有数据都进行过算法迭代后，称为一次epoch

path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))

# 数据
print('==> Preparing data..')
# train、test图像预处理和增强
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),  # 随机剪裁
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),  # 对图片进行上下左右以及中心裁剪，然后全部翻转（水平或者垂直），获得10张图片，返回一个4D-tensor
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])
# 加载train、test数据集
train_set = CK(split='Training', fold=opt.fold, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.bs, shuffle=True, num_workers=1)
test_set = CK(split='Testing', fold=opt.fold, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=5, shuffle=False, num_workers=1)

# 模型
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model == 'Resnet18':
    net = ResNet18()
elif opt.model == 'CNN':
    net = CNN()

if opt.resume:
    # 加载检查点
    print('==> Resuming from checkpoint..')  # 输出“从检查点恢复”
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(path, 'Test_model.t7'))  # 如果条件返回错误，则报错

    net.load_state_dict(checkpoint['net'])
    best_Test_acc = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch = best_Test_acc_epoch + 1
else:
    print('==> Building model..')

# 把模型参数传给optim前需要调用cuda（）
if use_cuda:
    net.cuda()

# 初始化loss和优化函数
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)  # 使用SGD优化，设置学习率、动量、权重衰减


# 训练
def train(epoch, start_time):
    t = datetime.datetime.now()  # 从系统当前时间开始
    t_show = datetime.datetime.strftime(t, '%Y-%m-%d %H:%M:%S')  # 格式化显示时间
    print('Epoch: %d\t%s\t%s' % (epoch, t_show, utils.cal_run_time(start_time)))  # 输出epoch，时间，运行时间
    global Train_acc  # 记录准确率
    net.train()  # 网络训练
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # 设置衰减率
    else:
        current_lr = opt.lr
    print('learning_rate: %.6f' % current_lr)

    # 迭代训练
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()  # 优化器置零
        inputs, targets = Variable(inputs), Variable(targets)  # 准备tensor的训练数据和标签
        outputs = net(inputs)  # 前向传播计算网络结构的输出结果
        loss = criterion(outputs, targets)  # 计算损失函数
        loss.backward()  # 反向传播
        utils.clip_gradient(optimizer, 0.1)  # 设置梯度阈值预防梯度爆炸
        optimizer.step()  # 更新所有的参数，进行单次优化

        # 统计预测信息
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (train_loss / (batch_idx + 1), 100 * float(correct) / total, correct, total))

    Train_acc = 100 * float(correct) / total  # 训练正确率


# 测试
def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    # 初始化
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):  # 遍历测试数据集并把他组合为一个索引序列
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)  # fuse batch size and ncrops

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)  # 前向传播计算网络结构的输出结果
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops

        loss = criterion(outputs_avg, targets)  # 计算损失函数
        # 统计预测信息
        PrivateTest_loss += loss.data
        _, predicted = torch.max(outputs_avg.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PrivateTest_loss / (batch_idx + 1), 100 * float(correct) / total, correct, total))
    # save checkpoint.
    Test_acc = 100 * float(correct) / total  # 计算测试集的正确率

    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {'net': net.state_dict() if use_cuda else net,
                 'best_Test_acc': Test_acc,
                 'best_Test_acc_epoch': epoch,  # 字典  关键字：值
                 }
        if not os.path.isdir(opt.dataset + '_' + opt.model):
            os.mkdir(opt.dataset + '_' + opt.model)
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'Test_model.t7'))  # 路径拼接和保存
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch


start_time = time.time()
for epoch in range(start_epoch, total_epoch):
    train(epoch, start_time)
    test(epoch)

print("best_Test_acc: %0.3f" % best_Test_acc)  # 输出测试集识别的最高正确率
print("best_Test_acc_epoch: %d" % best_Test_acc_epoch)  # 输出测试集识别最高正确率的epoch
