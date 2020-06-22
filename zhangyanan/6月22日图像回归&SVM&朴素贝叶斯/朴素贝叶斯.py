# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月22日
# 修改日期：
# 修改人：
# 修改内容：
'''
 主要参数 alpha平滑参数，越小越容易过拟合，越大越容易欠拟合
 使用交叉验证
 使用精确率 召回率 和 F1得分
 使用混淆矩阵
'''

import numpy as np
import pandas as pd

data = pd.read_csv('Tweets.csv')
print(data.head())

data = data[['airline_sentiment', 'text']]
data.airline_sentiment.str.replace('neutral', 'gg')
print(data.airline_sentiment)

import re
token = re.compile(r'[A-Za-z]+|[!?.:,()]')  # 提取出全部的英文字母和常用的标点符号，其他的都不要

def extract_text(text):  # 传入text，可以提取出全部的英文字母和标点，再转为小写
    new_text = token.findall(text)  # 提取出全部的英文字母和标点
    new_text = ' '.join([x.lower() for x in new_text])  # 转成小写
    return new_text

x = data.text.apply(extract_text)  # 给data的text应用extract_text函数
y = data.airline_sentiment


# 先划分，再进行向量化，因为如果先向量化，再划分，就相当于提前知道了全部数据
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(x_train.shape, x_test.shape)

# 向量化处理
from sklearn.feature_extraction.text import TfidfVectorizer
# 如何被分词 ngram_range=(1, 3) 表示1-3这样分
#stop_words='english' 以英语作为暂停词
#min_df=3 出现频率小于3的就删除
vect = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', min_df=3)

x_train_vect = vect.fit_transform(x_train)  # 转换
x_test_vect = vect.transform(x_test)  # x不用fit_transform  只需要transform

from sklearn.naive_bayes import MultinomialNB  # 引入朴素贝叶斯
model = MultinomialNB(alpha=0.00001)
model.fit(x_train_vect,y_train)
print(model.score(x_train_vect,y_train))
print(model.score(x_test_vect, y_test))



from sklearn.model_selection import GridSearchCV  # 引入交叉验证，且允许你规定参数的范围，从而寻找到最合适的参数


values = np.linspace(0, 0.001, 50)

param_grid = {'alpha': values}  # 规定参数的范围，从而寻找到最合适的参数
model = GridSearchCV(MultinomialNB(), param_grid, cv=5)
model.fit(x_train_vect, y_train)
print(model.best_params_)
print(model.best_score_)



from sklearn.metrics import  classification_report  # 引入评价指标
model = MultinomialNB(alpha=0.00001)
model.fit(x_train_vect,y_train)

# 预测
pred = model.predict(x_test_vect)
# 看预测情况
print(classification_report(y_test, pred))

'''
              precision    recall  f1-score   support

    negative       0.76      0.94      0.84      2280
     neutral       0.61      0.34      0.43       804
    positive       0.74      0.50      0.60       576

    accuracy                           0.74      3660
   macro avg       0.70      0.59      0.62      3660
weighted avg       0.72      0.74      0.71      3660
'''

from sklearn.metrics import confusion_matrix  # 引入混淆矩阵
cm = confusion_matrix(y_test, pred)
print(cm)
'''混淆矩阵
[[2161  104   40]
 [ 386  281   74]
 [ 222   69  323]]

'''