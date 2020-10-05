# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/9/25 002523:53
# 文件名称：voc_annotation
# 开发工具：PyCharm
import xml.etree.ElementTree as ET
import os
# 这个主要生成200_test.txt,2007_train.txt,2007_cal.txt

# 训练，验证，测试
sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# 20个类
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
           'sofa', 'train', 'tvmonitor']


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id))
    tree = ET.parse(in_file)  # 得到树
    root = tree.getroot()  # 得到根节点

    for obj in root.iter('object'):  # 查找标签是object的所有标签
        difficult = obj.find('difficult').text  # 查找object标签下的difficult标签
        cls = obj.find('name').text  # 获得物体的名称，object标签下的name标签
        if cls not in classes or int(difficult) == 1:  # 如果不是物体（背景）
            continue  # 进行下一个物体
        cls_id = classes.index(cls)  # 获取物体的索引
        xmlbox = obj.find('bndbox')  # 准备获取2个坐标（左上和右下）
        # 获取物体的坐标，
        b = (xmlbox.find('xmin').text, xmlbox.find('ymin').text, xmlbox.find('xmax').text, xmlbox.find('ymax').text)
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
wd = os.getcwd() # 得到当前目录
for year,image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt' % (year, image_set)).read().strip().split() # 进行空格分隔
    list_file = open('%s_%s.txt'%(year,image_set),'w') # 创造类似voc2007_train的txt文件
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd,year,image_id))
        convert_annotation(year,image_id,list_file)
        list_file.write('\n')
    list_file.close()

