# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月22日
# 修改日期：
# 修改人：
# 修改内容：
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("credit-a.csv", header=None)
print(data.head())
print(data.tail())

from sklearn.model_selection import train_test_split
x = data[data.columns[:-1]]
y = data[15].replace(-1, 0)
x_train, x_test, y_train, y_test = train_test_split(x, y)

from sklearn import preprocessing  # 预处理模块

# 归一化
scaler = preprocessing.StandardScaler().fit(x_train)  # 只使用一个标准化
# 统一保准化可以使x_train和x_test在一起标准化
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.svm import SVC
model = SVC(kernel="poly", degree=3, C=5)  # 核函数使用多项式核函数，degree为3，惩罚系数为5
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

model2 = SVC(kernel="rbf", gamma=0.5, C=5)  # 核函数使用高斯核函数
model2.fit(x_train, y_train)
print(model2.score(x_test, y_test))

