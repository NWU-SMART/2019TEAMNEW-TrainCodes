# -*- coding: utf-8 -*-
# @Time: 2020/7/2 15:25
# @Author: wangshengkang
# @Software: PyCharm
import os
from shutil import copyfile

# 数据集路径
download_path = 'D:\datasets\zhengzhedong\Market-1501-v15.09.15\Market-1501-v15.09.15'
'''
Market1501数据集内容:
    bounding_box_test  19732张测试集图片，750人。前缀为 0000 表示在提取这 750 人的过程中DPM检测错的图（图片可能为局部图片）（可能与query是同一个人），-1 表示检测出来其他人的图（不在这 750 人中）
    bounding_box_train  12936张训练集图片，751人。
    gt_bbox  手工标注的bounding box，用于判断DPM检测的bounding box是不是一个好的box。25259张图片，图片与train和test中的1501个行人有关。用来区分good，junk，distractors
    gt_query  matlab格式，用于判断一个query的哪些图片是好的匹配（同一个人不同摄像头的图像）和不好的匹配（同一个人同一个摄像头的图像或非同一个人的图像）。与3368个query有关，里面有good，junk，在performance evaluation的时候用
    query  3368张query图片，750人，对应的gallery为bounding_box_test
    readme.txt
    
经过prepare.py处理后，在pytorch文件夹中生成的新的数据集形式：
    gallery             gallery也就是test
    multi-query         gt_box里的数据
    query               query
    train               train
    train_all           train+val
    val                 val
'''
# 如果数据集地址不对，提示
if not os.path.isdir(download_path):
    print('please change the download_path')

# 数据处理后的保存地址
save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)  # 如果没有这个文件夹，创建一个新的
# --------------------------------------------------------------------------------
# query
query_path = download_path + '/query'
query_save_path = download_path + '/pytorch/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)  # 如果没有这个文件夹，创建一个新的

# os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
# os.walk() 方法是一个简单易用的文件、目录遍历器，可以帮助我们高效的处理文件、目录方面的事情。
# top -- 是你所要遍历的目录的地址, 返回的是一个三元组(root,dirs,files)。
# root 所指的是当前正在遍历的这个文件夹的本身的地址
# dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
# files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
# topdown --可选，为 True，则优先遍历 top 目录，否则优先遍历 top 的子目录(默认为开启)。
# 如果 topdown 参数为 True，walk 会遍历top文件夹，与top 文件夹中每一个子目录。
for root, dirs, files in os.walk(query_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':  # 如果后缀名不是jpg，跳出本次循环，不执行余下代码，继续下一次循环
            continue
        ID = name.split('_')  # 图片id，分成三块
        src_path = query_path + '/' + name  # 图片地址
        dst_path = query_save_path + '/' + ID[0]  # 用标签作为地址
        if not os.path.isdir(dst_path):  # 每个label文件夹的第一张图片时会创建新的文件夹，后面的就不用了
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)  # 将原始数据集复制到以label命名的新的文件夹

# ------------------------------------------------------------------------------------
# multi-query
query_path = download_path + '/gt_bbox'
# for dukemtmc-reid, we do not need multi-query
if os.path.isdir(query_path):
    query_save_path = download_path + '/pytorch/multi-query'
    if not os.path.isdir(query_save_path):
        os.mkdir(query_save_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:] == 'jpg':
                continue
            ID = name.split('_')
            src_path = query_path + '/' + name
            dst_path = query_save_path + '/' + ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)

# -----------------------------------------------------------------------------------
# gallery
gallery_path = download_path + '/bounding_box_test'
gallery_save_path = download_path + '/pytorch/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for root, dirs, files in os.walk(gallery_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = gallery_path + '/' + name
        dst_path = gallery_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# -------------------------------------------------------------------------------------
# train_all
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# --------------------------------------------------------------------------------------
# train
# val
train_path = download_path + '/bounding_box_train'
train_save_path = download_path + '/pytorch/train'
val_save_path = download_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
    os.mkdir(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name.split('_')
        src_path = train_path + '/' + name
        dst_path = train_save_path + '/' + ID[0]
        # 每个label的第一张图片会创建val新的文件夹，并且复制到val中去
        # 如果不是第一张图片则会直接跳过if not，复制到train中去
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
            dst_path = val_save_path + '/' + ID[0]  # first image is used as val image
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/' + name)
