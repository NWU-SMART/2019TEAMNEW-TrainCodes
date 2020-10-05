# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/9/25 002522:51
# 文件名称：voc2ssd
# 开发工具：PyCharm
import os
import random
'''
代码主要就是写一个test.txt。把图片的名字写进去，train.txt和其他的已经有了，所以就不写其他的了
'''



xml_file_path = 'Annotations'
save_base_path = 'ImageSets/Main/'

trainval_percent = 1
train_percent = 1

# Annotations 目录下所有文件的名字
temp_xml = os.listdir(xml_file_path)
total_xml = [] # 总共的xml个数

for xml in temp_xml:
    if xml.endswith(".xml"): # 如果文件是.xml结尾的
        total_xml.append(xml) # 向total_xml中添加

num = len(total_xml) # 此处num相当于图片的数量
list = range(num)


test = random.sample(list,int(num*0.1))


ftest = open(os.path.join(save_base_path,'test.txt'),'w')
for i in list: # 对于每个xml的索引
    name = total_xml[i][:-4]+'\n' # \n表示空格
    if i in test:
        ftest.write(name)

ftest.close()







