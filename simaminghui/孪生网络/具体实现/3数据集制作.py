# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/22 002217:31
# 文件名称：3数据集制作
# 开发工具：PyCharm


"""
代码为制作数据集，相同数字标签为1，不同数字标签为0，有了数据集才能训练
数据集为x1[i]，x2[i]，Y[i]。若x1[i]和x2[i]相同（数字相同）则Y[i]为1,否则Y[i]为0
"""
import numpy as np
import random
path = 'D:\DataList\mnist\mnist.npz'
f = np.load(path)
x1, y_train = f['x_train'], f['y_train']

x2 = []

# ---------------------制作数据集-------------------
'''
让数据集尽量平衡，
'''
Y = np.zeros([60000, 1])  # 60000,1全为0的矩阵
for i in range(60000):
    # 如果两个图像的标签相同，则两个图像相似度为100%，即1。如果标签不相同，则不相似，Y[i]不用改变，为0.
    id = random.randint(0, 60000 - 1)  # 60000-1是因为y_train[60000]不存在
    # 9/10是不同的,奇数为不同
    if i%2==1: # 索引为奇数，表示两个数字不同
        while y_train[i] == y_train[id]: # 当两个数字相同时（1/10概率）
            id = random.randint(0, 60000 - 1)
    if i%2==0: # 索引为偶数，表示两个数字相同
        while y_train[i] != y_train[id]: # 当两个数字相同时（1/10概率）
            id = random.randint(0, 60000 - 1)
        Y[i] = 1
    x2.append(x1[id])

# 将x2转为array（此时x2为list）
x2 = np.array(x2)


# ---------------------制作数据集end-------------------


np.savez('siameseData.npz', x1=x1, x2=x2,Y=Y)

# 之后运行4代码实现
