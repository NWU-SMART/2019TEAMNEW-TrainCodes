# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月18日
# 修改日期：
# 修改人：
# 修改内容：决策树实现房价预测
'''
# 决策树是一种十分常用的分类方法，需要监管学习（有教师的Supervised Learning），监管学习就是给出一堆样本，每个样本都有一组属性和一个分类结果，
  也就是分类结果已知，那么通过学习这些样本得到一个决策树，这个决策树能够对新的数据给出正确的分类。

#  所以决策树的生成主要分以下两步，这两步通常通过学习已经知道分类结果的样本来实现。
   1. 节点的分裂：一般当一个节点所代表的属性无法给出判断时，则选择将这一节点分成2个
      子节点（如不是二叉树的情况会分成n个子节点）
   2. 阈值的确定：选择适当的阈值使得分类错误率最小 （Training Error）。
     比较常用的决策树有ID3，C4.5和CART（Classification And Regression Tree），CART的分类效果一般优于其他决策树。下面介绍具体步骤。
     ID3: 由增熵（Entropy）原理来决定那个做父节点，那个节点需要分裂。对于一组数据，熵越小说明分类结果越好。熵定义如下：
     Entropy＝- sum [p(x_i) * log2(P(x_i) ]
    其中p(x_i) 为x_i出现的概率。假如是2分类问题，当A类和B类各占50%的时候，
    Entropy = - （0.5*log_2( 0.5)+0.5*log_2( 0.5))= 1
    当只有A类，或只有B类的时候，
    Entropy= - （1*log_2( 1）+0）=0
    所以当Entropy最大为1的时候，是分类效果最差的状态，当它最小为0的时候，是完全分类的状态。因为熵等于零是理想状态，一般实际情况下，熵介于0和1之间。
    熵的不断最小化，实际上就是提高分类正确率的过程。
'''

'''感觉决策树好像更适合逻辑回归，不适合普通回归问题'''
# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包  sss panda是一个可数据预处理的包
# 2、房价训练数据导入
# 3、数据归一化
# 4、模型训练
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
from keras.preprocessing import sequence
from keras.models import Sequential#顺序模型
from keras.datasets import boston_housing
from keras.layers import Dense, Dropout#全连接层
from keras.utils import multi_gpu_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt#画图工具
import numpy as np#科学计算库
from sklearn.preprocessing import MinMaxScaler
import pandas as pd  #数据预处理的工具
from sklearn.linear_model import LassoCV

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
path = 'boston_housing.npz'
f = np.load(path)      # sss   numpy.load（）读取数据
# 404个训练，102个测试
# 训练数据
x_train=f['x'][:404]  # 下标0到下标403
y_train=f['y'][:404]
# 测试数据
x_valid=f['x'][404:]  # 下标404到下标505
y_valid=f['y'][404:]
f.close()   # 关闭文件
# 数据放到本地路径

# 转成DataFrame格式方便数据处理    sssDataFrame格式可理解为一张表
x_train_pd = pd.DataFrame(x_train)#Excel格式
y_train_pd = pd.DataFrame(y_train)
x_valid_pd = pd.DataFrame(x_valid)
y_valid_pd = pd.DataFrame(y_valid)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)


print(x_train_pd.info())
#  -------------------------- 2、房价训练和测试数据载入 -------------------------------


#  -------------------------- 3、数据归一化 -------------------------------
# 训练集归一化 归一化可以减少量纲不同带来的影响，使得不同特征之间具有可比性；这里用的是线性归一化，(x-xmin)/(xmax-xmin)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)

# 验证集归一化
min_max_scaler.fit(x_valid_pd)
x_valid = min_max_scaler.transform(x_valid_pd)

min_max_scaler.fit(y_valid_pd)
y_valid = min_max_scaler.transform(y_valid_pd)

#  -------------------------- 4、模型编译和训练   -------------------------------
'''这种方法是不设置超参数的'''
np.shape(x_train), np.shape(x_valid)
from sklearn.tree import DecisionTreeRegressor  # 引入决策树
model = DecisionTreeRegressor()  # 实例化模型

model.fit(x_train, y_train.astype('float'))

print(model.score(x_valid, y_valid))   # 测试集准确率
print(model.score(x_train, y_train))  # 训练集准确率


'''设置超参数'''
def m_score(depth):  # 定义一个改变决策树深度的函数
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_valid, y_valid)
    return train_score,test_score

depths = range(2,15)  # 深度在2-15之间改变
scores = [m_score(depth) for depth in depths]
print(scores)


