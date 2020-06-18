#=------------------2020.06.18------------------
#导入包
#导入数据
#数据分析和处理
#建模
#评估
#-------------------------------------------------导入包----------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.model_selection import cross_val_score
#import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
#-------------------------------------------------导入包----------------------------------

#-------------------------------------------------导入数据----------------------------------
traindata = pd.read_csv('D:\\360安全浏览器下载\\MobileFile\\train.csv')
# 了解数据具体信息
#traindata.info()
#-------------------------------------------------导入数据----------------------------------

#-------------------------------------------------数据分析----------------------------------
total = traindata.isnull().sum().sort_values(ascending=False)

percent = (traindata.isnull().sum()/len(traindata)).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])
#print(missing_data)
traindata=traindata.drop(missing_data[missing_data['percent']>0.8].index,axis=1)
#traindata.info()

#计算相关性
encoder=LabelEncoder()
traindata['房屋朝向']=encoder.fit_transform(traindata['房屋朝向'].values)
num_trainData = traindata.select_dtypes(include = ['int64', 'float64','int32'])
traindata_corr = traindata.corr()['Label'][:-1]
golden_feature_list = traindata_corr[abs(traindata_corr) > 0].\
    sort_values(ascending = False)
print("Below are {} correlated values with SalePrice:\n{}".format(len(golden_feature_list), golden_feature_list))
traindata=traindata.drop(['时间','距离','ID'],axis=1)
#traindata.info()
fig,ax=plt.subplots()
ax.scatter(x=traindata['房屋面积'],y=traindata['Label'],)
plt.ylabel('Price')
plt.xlabel('房屋面积')
#plt.show()
traindata=traindata.drop(traindata[traindata['房屋面积']>1500].index)
#traindata.info()
#相关系数最大的前三位做出他们的散点图，并且去除一些离群点


#补全数据
traindata['位置']=traindata['位置'].fillna(traindata['位置'].mean())
traindata['区']=traindata['区'].fillna(traindata['区'].mean())
traindata['卧室数量']=traindata['卧室数量'].fillna(traindata['卧室数量'].mean())
traindata['卫的数量']=traindata['卫的数量'].fillna(traindata['卫的数量'].mean())
traindata['厅的数量']=traindata['厅的数量'].fillna(traindata['厅的数量'].mean())
traindata['地铁站点']=traindata['地铁站点'].fillna(traindata['地铁站点'].mean())
traindata['地铁线路']=traindata['地铁线路'].fillna(traindata['地铁线路'].mean())
traindata['小区名']=traindata['小区名'].fillna(traindata['小区名'].mean())
traindata['小区房屋出租数量']=traindata['小区房屋出租数量'].fillna(traindata['小区房屋出租数量'].mean())
traindata['总楼层']=traindata['总楼层'].fillna(traindata['总楼层'].mean())
traindata['房屋朝向']=traindata['房屋朝向'].fillna(traindata['房屋朝向'].mean())
traindata['房屋面积']=traindata['房屋面积'].fillna(traindata['房屋面积'].mean())
traindata['楼层']=traindata['楼层'].fillna(traindata['楼层'].mean())
#traindata.info()
#目标值处理处于正态分布

#建立模型
fig=plt.figure(figsize=(12,5))

if traindata['位置'].skew()>0.75:
    traindata['位置']=np.log1p(traindata['位置'])
if traindata['区'].skew()>0.75:
    traindata['区']=np.log1p(traindata['区'])
if traindata['卧室数量'].skew()>0.75:
    traindata['卧室数量']=np.log1p(traindata['卧室数量'])
if traindata['卫的数量'].skew()>0.75:
    traindata['卫的数量']=np.log1p(traindata['卫的数量'])
if traindata['厅的数量'].skew()>0.75:
    traindata['厅的数量']=np.log1p(traindata['厅的数量'])
if traindata['地铁站点'].skew()>0.75:
    traindata['地铁站点']=np.log1p(traindata['地铁站点'])
if traindata['地铁线路'].skew()>0.75:
    traindata['地铁线路']=np.log1p(traindata['地铁线路'])
if traindata['小区名'].skew()>0.75:
    traindata['小区名']=np.log1p(traindata['小区名'])
if traindata['小区房屋出租数量'].skew()>0.75:
    traindata['小区房屋出租数量']=np.log1p(traindata['小区房屋出租数量'])
if traindata['总楼层'].skew()>0.75:
    traindata['总楼层']=np.log1p(traindata['总楼层'])
if traindata['房屋朝向'].skew()>0.75:
    traindata['房屋朝向']=np.log1p(traindata['房屋朝向'])
if traindata['房屋面积'].skew()>0.75:
    traindata['房屋面积']=np.log1p(traindata['房屋面积'])
if traindata['楼层'].skew()>0.75:
    traindata['楼层']=np.log1p(traindata['楼层'])
g1=sns.distplot(traindata['Label'],hist=True,
                label='skewness:{:.2f}'.format(traindata['Label'].skew()))
g1.legend()
g1.set(xlabel='price')
#plt.show()
g2=sns.distplot(np.log1p(traindata['Label']),
                hist=True,
                label='skewness:{:.2f}'.format(np.log1p(traindata['Label']).skew()))
g2.legend()
g2.set(xlabel='log(Price+1)')
#-------------------------------------------------数据分析----------------------------------
#plt.show()
traindata['Label']=np.log1p(traindata['Label'])
ytrain=traindata['Label']
xtrain=traindata.drop(['Label'],axis=1)
#训练模型
#-------------------------------------------------建模---------------------------------
models=[LinearRegression(),KNeighborsRegressor(),
        MLPRegressor(alpha=20),
        DecisionTreeRegressor(),RandomForestRegressor(),
        AdaBoostRegressor(),GradientBoostingRegressor(),BaggingRegressor()]
models_str=['LinearRegression','KNNRegressor','MLPRegressor','DecisionTree',
            'RandomForest','AdaBoost','GradientBoost','Bagging']
score_=[]
#-------------------------------------------------建模---------------------------------

#-------------------------------------------------评估--------------------------------
for name,model in zip(models_str,models):
    print('开始训练模型：'+name)
    model=model#建立模型
    model.fit(xtrain,ytrain)
    score=model.score(xtrain,ytrain)
    score_.append(str(score)[:5])
    print(name +' 得分:'+str(score))
lr= LinearRegression()
lr.fit(xtrain,ytrain)
print("LinearRegression交叉验证得分：",cross_val_score(lr,xtrain,ytrain,cv=3).mean())
lr= DecisionTreeRegressor()
lr.fit(xtrain,ytrain)
print("DecisionTreeRegressor交叉验证得分：",cross_val_score(lr,xtrain,ytrain,cv=3).mean())


