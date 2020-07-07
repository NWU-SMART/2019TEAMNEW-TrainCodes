# -*- coding: utf-8 -*-
# @Time: 2020/7/3 12:43
# @Author: wangshengkang
# @Software: PyCharm

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
from model import ft_net, ft_net_dense, ft_net_NAS, PCB
# from random_erasing import RandomErasing
import yaml
import math
from shutil import copyfile

version = torch.__version__
# Options
# --------
# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='Training')
# 调用add_argument()方法添加参数
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
# 名字可以随便起，用来保存模型
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='/data0/wangshengkang/datasets/Market-1501-v15.09.15/pytorch', type=str,
                    help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--use_NAS', action='store_true', help='use NAS')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')

# 使用parse_args()解析添加的参数
opt = parser.parse_args()

data_dir = opt.data_dir  # 数据存放地址
name = opt.name  # 模型名字，默认为ft_ResNet50
str_ids = opt.gpu_ids.split(',')  # 将字符串按逗号分开
gpu_ids = []  # 建立gpu list
for str_id in str_ids:
    gid = int(str_id)  # str转为int
    if gid >= 0:
        gpu_ids.append(gid)  # 将可用的gpu id加入list

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])  # 设置使用哪块gpu
    '''
    总的来说，大部分情况下，设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    一般来讲，应该遵循以下准则：
    如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
    如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    '''
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((256, 128), interpolation=3),  # 改变分辨率
    transforms.Pad(10),  # 填充
    transforms.RandomCrop((256, 128)),  # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 依据概率p对PIL图片进行水平翻转，p默认为0.5
    transforms.ToTensor(),  # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 chw
]

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose(transform_train_list),  # 将transforms列表里面的transform操作进行遍历
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:  # 如果使用全部的训练数据train_all
    train_all = '_all'

image_datasets = {}
# 用ImageFolder包装数据集
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

# dataloader读取train，validation数据集
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                              shuffle=True, num_workers=8, pin_memory=True)  # 8 workers may work faster
               for x in ['train', 'val']}
# 数据长度
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# 子文件夹名，也就是label
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()  # 看gpu是否可用

since = time.time()  # 获取当前时间
inputs, classes = next(iter(dataloaders['train']))  # 迭代读取
print(time.time() - since)  # 打印出代码运行时间

y_loss = {'train': [], 'val': []}  # loss history 创建损失字典
y_err = {}
y_err['train'] = []
y_err['val'] = []


# 训练的函数
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()  # 获取系统时间

    # best_model_wts = model.state_dict()
    # best_acc = 0.0
    warm_up = 0.1  # We start from the 0.1*lrRate
    # round返回浮点数x的四舍五入值。
    warm_iteration = round(dataset_sizes['train'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    for epoch in range(num_epochs):  # 开始训练
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))  # 打印epoch数
        print('-' * 10)  # 打印分割线

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data  # 获得图片和标签
                now_batch_size, c, h, w = inputs.shape  # 获取N,C,H,W
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue  # 跳出去，进行下一次循环
                # print(inputs.shape)
                # wrap them in Variable
                if use_gpu:  # 将数据和标签用Variable包装，并且放到gpu中
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)  # 求每一行的最大值，以及最大值的列标，列标作为标签
                loss = criterion(outputs, labels)  # 计算损失

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    loss.backward()  # backward
                    optimizer.step()  # optimizer

                # statistics
                # if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                running_loss += loss.item() * now_batch_size
                # else:  # for the old version like 0.3.0 and 0.3.1
                # running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # 打印每个epoch的损失和准确率
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
            # deep copy the model
            if phase == 'val':
                last_model_wts = model.state_dict()  # model.state_dict()其实返回的是一个OrderDict，存储了网络结构的名字和对应的参数
                if epoch % 10 == 9:  # 每十次epoch保存一次网络
                    save_network(model, epoch)

        time_elapsed = time.time() - since  # 所用的时间
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()  # 打印空行

    time_elapsed = time.time() - since  # 总消耗时间
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(last_model_wts)  # 加载模型参数
    save_network(model, 'last')
    return model


# Save model
# ---------------------------
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label  # 模型名字
    save_path = os.path.join('./model', name, save_filename)  # 模型路径
    torch.save(network.cpu().state_dict(), save_path)  # 保存模型
    if torch.cuda.is_available():
        network.cuda(gpu_ids[0])


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

# 调用模型
model = ft_net(len(class_names), opt.droprate, opt.stride)

opt.nclasses = len(class_names)  # 类的数量，也就是行人ID数量

print(model)  # 打印model网络结构

ignored_params = list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
# 优化器为SGD
optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.1 * opt.lr},
    {'params': model.classifier.parameters(), 'lr': opt.lr}
], weight_decay=5e-4, momentum=0.9, nesterov=True)

# Decay LR by a factor of 0.1 every 40 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU.
#
dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)
# record every run 将本次训练和模型的文件备份保存
copyfile('./train.py', dir_name + '/train.py')
copyfile('./model.py', dir_name + '/model.py')

# 'w'表示打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
# save opts 将本次的参数保存下来
with open('%s/opts.yaml' % dir_name, 'w') as fp:
    # dump()将一个python对象生成为yaml文档，
    yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
model = model.cuda()

# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 调用训练函数
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    num_epochs=60)
