# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/29 002910:13
# 文件名称：__init__.py
# 开发工具：PyCharm

# 数据路径
import random

import pandas

path = 'D:\DataList\prostate-cancer\Prostate_Cancer.csv'

# 转换为矩阵
data = pandas.read_csv(path).values
print(data[5])
# 打乱顺序
random.shuffle(data)
print(data[5])
n = len(data) // 3
# 分为测试集和训练接
test_data = data[:n]  # 1/3做测试集
train_data = data[n:]  # 2/3做训练集


# 算出两点距离，采用欧几里得距离
def distance(x, y):  # x与y的距离
    sum = 0
    for i in range(2, 10):
        sum += (float(x[i]) - float(y[i])) ** 2
    # 开方
    return sum ** 0.5


def MorB(x):  # 查看x是M还是B
    res = []
    # 得到x与所有训练集的距离
    for item in train_data:
        res.append({"result": item[1], "distance": distance(x, item)})
    # 进行排序（升序）
    res = sorted(res, key=lambda item: item['distance'])
    # 取前K个
    k = 5
    res = res[:k]
    # 开始M和B权均为0
    M, B = 0, 0
    # 进行加权平均
    sum = 0
    for item in res:
        sum = sum + item['distance']
    for item in res:
        if item['result'] == 'M':
            M += 1 - item['distance'] / sum
        else:
            B += 1 - item['distance'] / sum

    if M > B:
        return 'M'
    else:
        return 'B'




# 查看准确率
acc = 0
for item in test_data:
    res = MorB(item)
    if res==item[1]:
        acc +=1

print('准确个数为：',acc)
print('总共为：',len(test_data))
# 可达到80+准确率
print("准确率为：",acc*100/len(test_data))