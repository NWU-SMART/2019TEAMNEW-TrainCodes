# ----------------开发者信息--------------------------------#
# 开发人员：司马明辉
# 开发日期：2020/7/7 000721:10
# 文件名称：test
# 开发工具：PyCharm
import numpy
from tqdm import tqdm

# s = numpy.random.uniform(0, 1, size=[5, 5])
# print(s)
#
# y = numpy.zeros([10, 3])
# y[:5,2] = 1
# print(y)

x = numpy.array([[1,2,3],[4,5,6],[7,8,9],[4,8,2]])
y = numpy.random.randint(0,4,3)

print(x[y])

x = numpy.array([1,2,3])
y = numpy.array([4,5,6])
print(0.5*numpy.add(x,y))