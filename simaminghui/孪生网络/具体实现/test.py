# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/22 002215:27
# 文件名称：test
# 开发工具：PyCharm

import keras
import numpy as np
import matplotlib.pyplot as plt

'''
frist（）中的数据集是siameseData.npz中的，即没有验证集，
second（）中有创建了一个数据集，当做验证集（siameseData2.npz），也就是在运行一遍3数据集制作
的代码，把里面的siameseData.npz改为siameseData2.npz便可
'''
def first():
    path = 'siameseData.npz'
    f = np.load(path)

    x1, x2, Y = f["x1"], f['x2'], f['Y']
    x = []
    y = []
    id = 0
    # 找10个数字相同的组成数据集，然后测试，理论输出全是1，（如有意外，纯属理论不够）
    for i in range(len(Y)):
        if id<10:
            if Y[i] == 1:
                x.append(x1[i])
                y.append(x2[i])
                id = id+1

    x = np.asarray(x)
    y = np.asarray(y)
    x = x.reshape(10,28,28,1)
    y = y.reshape(10,28,28,1)


    model = keras.models.load_model('siamese_model2.h5')
    print(model.predict([x,y]))


# 可以在制作一个测试集
def second():
    path = 'siameseData2.npz'
    f = np.load(path)

    x1, x2, Y = f["x1"], f['x2'], f['Y']
    # 数据处理
    x1 = x1.reshape(60000,28,28,1)
    x2 = x2.reshape(60000,28,28,1)
    # 查看准确率
    model = keras.models.load_model('siamese_model2.h5')
    print(model.evaluate([x1,x2],Y))

second() # 准确率大概97.49%