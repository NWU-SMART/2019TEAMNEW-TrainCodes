import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
import xgboost as xgb
from sklearn.linear_model import LinearRegression 


traindata = pd.read_csv('D:\\Users\\linglly\\Desktop\\train.csv')
testdata = pd.read_csv('D:\\Users\\linglly\\Desktop\\test_noLabel.csv')

# 了解数据具体信息
traindata.info()

total = traindata.isnull().sum().sort_values(ascending=False)
percent = (traindata.isnull().sum()/traindata.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

num_trainData = traindata.select_dtypes(include = ['int64', 'float64'])


traindata_corr = num_trainData.corr()['price'][:-1]
golden_feature_list = traindata_corr[abs(traindata_corr) > 0].sort_values(ascending = False)
print("Below are {} correlated values with SalePrice:\n{}".format(len(golden_feature_list), golden_feature_list))


traindata_corrheatmap = num_trainData.corr()
cols = traindata_corrheatmap.nlargest(10, 'price')['price'].index
cm = np.corrcoef(num_trainData[cols].values.T)
sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', 
            annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)



# 查看'pice'的分布情况
traindata['price'].describe()

sns.distplot(traindata['price'], color='b', bins=100)
probmap = stats.probplot(traindata['price'], plot=plt)
sns.distplot(np.log(traindata['price']), color='r', bins=100)
res = stats.probplot(np.log(traindata['price']), plot=plt)
res = stats.probplot(traindata['area_house'], plot=plt)
sns.distplot(traindata['area_house'], color='b', bins=100)
res = stats.probplot(np.log(traindata['area_house']), plot=plt)

#把这些房屋特征和'price'的相关性可视化。
traindata.plot.scatter(x='area_house', y='price')

traindata.plot.scatter(x='floorage', y='price')

traindata.plot.scatter(x='num_bathroom', y='price')

traindata.plot.scatter(x='area_basement', y='price')

#sns.boxplot(x='rating', y='price', data=traindata)

#sns.boxplot(x='num_bedroom ', y='price', data=traindata)

#sns.boxplot(x='latitude', y='price', data=traindata)

#sns.boxplot(x='floor', y='price', data=traindata)

#sns.boxplot(x='year_repair', y='price', data=traindata)

#sns.boxplot(x='year_built', y='price', data=traindata)

#sns.boxplot(x='area_parking', y='price', data=traindata)

finaldata = traindata.filter(['area_house','rating', 'floorage', 'num_bathroom', 
                              'area_basement', 'num_bedroom', 'latitude', 'floor', 
                              'year_repair','area_parking', 'year_built ','price'], axis=1)
finaltest = testdata.filter(['area_house','rating', 'floorage', 'num_bathroom',
                             'area_basement', 'num_bedroom', 'latitude', 'floor', 
                             'year_repair','area_parking','year_built '], axis=1)

xtrain = finaldata.iloc[:,0:10].values
ytrain = finaldata['price']
xtest = finaltest.values

# 创建预测模型
regr = xgb.XGBRegressor()
regr.fit(xtrain, ytrain)

# 计算XGBoost模型得分0.91989295502373492
regr.score(xtrain, ytrain)
print(regr.score(xtrain, ytrain))
# 用XGBoost运行模型
y_pred = regr.predict(xtrain)

# 用测试集预测房屋价格
y_test = regr.predict(xtest)


# 使用线性回归模型
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

# 计算模型得分0.8769201701134044
regressor.score(xtrain,ytrain)
print(regressor.score(xtrain,ytrain))
# 运行模型
ytrainpred = regressor.predict(xtrain)

# 预测价格
ytestpred = regressor.predict(xtest)

# 计算两种预测结果的平均值,使用exp将对数转换
finalpred = (y_test + ytestpred) / 2
#finalpred = np.exp(finalpred)


pred_df = pd.DataFrame(finalpred, index=testdata["ID"], columns=["price"])
pred_df.to_csv('Predicting_house_price_output.csv', header=True, index_label='ID')

