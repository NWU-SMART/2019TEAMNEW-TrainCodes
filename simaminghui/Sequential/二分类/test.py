# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/6/18 001810:04
# 文件名称：test
# 开发工具：PyCharm
import numpy as np

x = np.array([[1, 6, 1],
              [85, 9, 6, 1], [7, 8, 5]])
results = np.zeros((len(x), 86))
print(results)
for i, sequence in enumerate(x):
    print('我是i：', i, '我是sequence:', sequence)
    results[i, sequence] = 1
print(results)
