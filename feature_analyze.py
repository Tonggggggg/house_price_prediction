
# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

alldata = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']), ignore_index=True)


alldata.shape

explore = train.describe(include = 'all').T
explore['null'] = len(train) - explore['count']
explore.insert(0,'dtype',train.dtypes)
explore.T.to_csv('explore1.csv')


explore.head()


explore = alldata.describe(include = 'all').T
explore['null'] = len(alldata) - explore['count']
explore.insert(0,'dtype',alldata.dtypes) 
explore.T.to_csv('explore2.csv')

explore.head()


# 相关图
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.9, square=True)

cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

#查看影响最终价格的十个变量
k = 10 
plt.figure(figsize=(12,9))
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


Corr = train.corr()
Corr[Corr['SalePrice']>0.5]


#scatterplot 绘制散点图矩阵注意：多变量作图数据中不能有空值，否则出错
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show();

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt','YearRemodAdd','TotRmsAbvGrd','GarageArea']
sns.pairplot(train[cols], size = 2.5)
plt.show();


train['SalePrice'].describe()


# 画直方图,且查看数据是否符合正态分布
# 直方图和正态概率图
from scipy.stats import norm
sns.distplot(train['SalePrice'],fit=norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
 
##  由图像可知，图像的非正态分布


print("Skewness: %f" %train['SalePrice'].skew()) #偏度
print("Kurtosis: %f" %train['SalePrice'].kurt()) #峰度


# 研究SalePrice和GrLivArea的关系
data1 = pd.concat([train['SalePrice'], train['GrLivArea']], axis=1)
data1.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
 
# 直方图和正态概率图，查看是否正态分布

fig = plt.figure()
sns.distplot(train['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['GrLivArea'], plot=plt)

##  由散点图可知，图像的右下角存在两个异常值，建议去除；图像非正态分布


# 研究SalePrice和TotalBsmtSF的关系
# TotalBsmtSF代表地下室有多少平方米
data1 = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)
data1.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0,800000));
 
# 直方图和正态概率图，查看是否正态分布
fig = plt.figure()
sns.distplot(train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(train['TotalBsmtSF'], plot=plt)
 
##  由散点图可知，图像的右下角存在1个异常值，建议去除该记录；图像非正态分布


# 研究SalePrice和OverallQual的关系

data2 = pd.concat([train['SalePrice'], train['OverallQual']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='OverallQual', y="SalePrice", data=data2)
fig.axis(ymin=0, ymax=800000);



# 查看不同月份的房子的销售量

print(train.groupby('MoSold')['SalePrice'].count())
sns.countplot(x='MoSold',data=train)

