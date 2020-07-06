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
# from PIL import Image
import time
import os
# from model import ft_net, ft_net_dense, ft_net_NAS, PCB
# from random_erasing import RandomErasing
# import yaml
import math
from shutil import copyfile

version = torch.__version__
# Options
# --------
# 创建 ArgumentParser() 对象
parser = argparse.ArgumentParser(description='Training')
# 调用add_argument()方法添加参数
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='../Market/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--stride', default=2, type=int, help='stride')
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