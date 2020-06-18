# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月18日
# 修改日期：
# 修改人：
# 修改内容：随机森林实现房价预测
'''
#随机森林是一种灵活的、便于使用的机器学习算法，即使没有超参数调整，大多数情况下也会带来好的结果。它可以用来进行分类和回归任务。

随机森林的工作原理如下：

1. 从数据集（表）中随机选择k个特征（列），共m个特征（其中k小于等于m）。然后根据这k个特征建立决策树。

2. 重复n次，这k个特性经过不同随机组合建立起来n棵决策树（或者是数据的不同随机样本，称为自助法样本）。

3. 对每个决策树都传递随机变量来预测结果。存储所有预测的结果（目标），你就可以从n棵决策树中得到n种结果。

4. 计算每个预测目标的得票数再选择模式（最常见的目标变量）。换句话说，将得到高票数的预测目标作为随机森林算法的最终预测。

*针对回归问题，随机森林中的决策树会预测Y的值（输出值）。通过随机森林中所有决策树预测值的平均值计算得出最终预测值。而针对分类问题，随机森林中的每棵决策树会预测最新数据属于哪个分类。最终，哪一分类被选择最多，就预测这个最新数据属于哪一分类。
'''
#  -------------------------- 1、导入需要包 -------------------------------
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

#  -------------------------- 2、房价训练和测试数据载入 -------------------------------
path = 'boston_housing.npz'
f = np.load(path)
x_train=f['x'][:]  # 下标0到下标403
y_train=f['y'][:]

f.close()   # 关闭文件
# 数据放到本地路径

# 转成DataFrame格式方便数据处理
x_train_pd = pd.DataFrame(x_train)#Excel格式
y_train_pd = pd.DataFrame(y_train)
print(x_train_pd.head(5))  # 输出 房屋训练数据的x (前5个)
print('-------------------')
print(y_train_pd.head(5))  # 输出 房屋训练数据的y (前5个)

#  -------------------------- 3、数据归一化 -------------------------------
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(x_train_pd)
x_train = min_max_scaler.transform(x_train_pd)

min_max_scaler.fit(y_train_pd)
y_train = min_max_scaler.transform(y_train_pd)


#  -------------------------- 4、模型编译和训练   -------------------------------
'''不改变超参数'''
#from sklearn.model_selection import train_test_split  # 导入划分包
#x_train, x_test, y_train, y_test = train_test_split(x_train_pd, y_train_pd, test_size=0.2)  # 划分训练集和测试集
#from sklearn.ensemble import RandomForestRegressor  # 随机森林
# 100棵树，4线程
#odel = RandomForestRegressor(n_estimators=100, n_jobs=4)
#model.fit(x_train, y_train)
#print(model.score(x_test, y_test))
#print(model.feature_importances)  # 看哪些特征比较重要


'''改变超参数，并使用交叉验证'''
from sklearn.ensemble import RandomForestRegressor  # 随机森林
from sklearn.model_selection import GridSearchCV  # 交叉验证
n_estimators = range(80, 130) # 定义80-130棵树
param_grid = {'n_estimators': n_estimators}  # n_estimators，树的个数
model = GridSearchCV(RandomForestRegressor(), param_grid, cv=5)  # 交叉验证划分为5份
model.fit(x_train_pd, y_train_pd)
print(model.best_params_)  # 打印超参数
print(model.best_score_)



