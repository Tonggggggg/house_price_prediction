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

alldata = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],  test.loc[:, 'MSSubClass':'SaleCondition']), ignore_index=True)


#################################缺失值处理###############################

def missing_values(alldata):
    # 统计所有数据中每列的na个数
    alldata_na = pd.DataFrame(alldata.isnull().sum(), columns={'missingNum'})
    alldata_na.head()

    # 计算每列的缺失值比率
    alldata_na['missingRatio'] = alldata_na['missingNum']/len(alldata)*100
    # 计算每列的非缺失值个数
    alldata_na['existNum'] = len(alldata) - alldata_na['missingNum']
    # 统计train data中每列的非缺失值个数    
    alldata_na['train_notna'] = len(train) - train.isnull().sum()
    # 计算test data中每列的非缺失值个数
    alldata_na['test_notna'] = alldata_na['existNum'] - alldata_na['train_notna'] 
    alldata_na['dtype'] = alldata.dtypes

    #按照missingNum将index重新排序
    alldata_na = alldata_na[alldata_na['missingNum']>0].reset_index().sort_values(by=['missingNum','index'],ascending=[False,True])
    alldata_na.set_index('index',inplace=True)
    return alldata_na
 
alldata_na = missing_values(alldata)

######## 处理pool的相关空值 ########

# 查看各个poolQC的分布情况
print(alldata['PoolQC'].value_counts())
# PoolArea的均值
poolqc = alldata.groupby('PoolQC')['PoolArea'].mean()
print('不同Poolqc的PoolArea的均值\n',poolqc)

# 查看有PoolArea数据但是没有poolQC的数据
poolqcna = alldata[(alldata['PoolQC'].isnull())& (alldata['PoolArea']!=0)][['PoolQC','PoolArea']]
print('查看有PoolArea数据但是没有PoolQC的数据\n',poolqcna)

# 查看无PoolArea数据但是有poolQC的数据
poolareana = alldata[(alldata['PoolQC'].notnull()) & (alldata['PoolArea']==0)][['PoolQC','PoolArea']]
print('查看无PoolArea数据但是有PoolQC的数据\n',poolareana)



######## 处理garage的相关空值 ########

# 找出所有Garage前缀的属性
a = pd.Series(alldata.columns)
GarageList = a[a.str.contains('Garage')].values
print(GarageList)

# -step1 GarageYrBlt	车库建造年份
print(alldata_na.ix[GarageList,:])


# -step2:检查GarageArea、GarageCars均为0的 ##(待处理)其他类别列的空值均填充“none”，数值列填“0”
#没车库，没车位
print(len(alldata[(alldata['GarageArea']==0) & (alldata['GarageCars']==0)]))# 157
#有车库，没车位
print(len(alldata[(alldata['GarageArea']!=0) & (alldata['GarageCars'].isnull==True)])) # 0
 
# 'GarageYrBlt'到后来与年份一起处理，也有空值


##### 找出所有Bsmt前缀的属性
a = pd.Series(alldata.columns)
BsmtList = a[a.str.contains('Bsmt')].values
print(BsmtList)
allBsmtNa = alldata_na.ix[BsmtList,:]
print(allBsmtNa)

condition2 = (alldata['BsmtQual'].isnull()) & (alldata['BsmtExposure'].notnull())
alldata[condition2][BsmtList]
# 通过研究发现，BsmtQual为空时，有两行数据其他值不为空，填充方法与condition1类似


condition = (alldata['BsmtExposure'].isnull()) & (alldata['BsmtCond'].notnull())
alldata[condition][BsmtList]
# 通过研究发现，BsmtExposure为空时，有三行数据其他值不为空，取众数填充
condition1 = (alldata['BsmtCond'].isnull()) & (alldata['BsmtExposure'].notnull())
print(len(alldata[alldata['BsmtCond']==alldata['BsmtQual']])) 
alldata[condition1][BsmtList]
# 通过研究发现，BsmtCond为空时，有三行数据其他值不为空# 有1265个值的
# BsmtQual == BsmtCond，所以对应填充


# 其他剩下的字段考虑数值型空值填0，标称型空值填none
print(alldata['BsmtFinSF1'].value_counts().head(5))# 空值填0
 
print(alldata['BsmtFinSF2'].value_counts().head(5))# 空值填0
 
print(alldata['BsmtFullBath'].value_counts().head(5))# 空值填0
 
print(alldata['BsmtHalfBath'].value_counts().head(5))# 空值填0
 
print(alldata['BsmtFinType1'].value_counts().head(5)) # 空值填Unf
 
print(alldata['BsmtFinType2'].value_counts().head(5)) # 空值填Unf


print(alldata[['MasVnrType', 'MasVnrArea']].isnull().sum())
 
print(len(alldata[(alldata['MasVnrType'].isnull())& (alldata['MasVnrArea'].isnull())])) # 23
 
print(len(alldata[(alldata['MasVnrType'].isnull())& (alldata['MasVnrArea'].notnull())])) # 1
 
print(len(alldata[(alldata['MasVnrType'].notnull())& (alldata['MasVnrArea'].isnull())])) # 0
 
print(alldata['MasVnrType'].value_counts())
 
MasVnrM = alldata.groupby('MasVnrType')['MasVnrArea'].median()
print('不同type的MasVnrArea的均值\n',MasVnrM)
mtypena = alldata[(alldata['MasVnrType'].isnull())& (alldata['MasVnrArea'].notnull())][['MasVnrType','MasVnrArea']]
print('MasVnrtype为空而MasVnrArea不为空\n',mtypena)
# 由此可知，由一条数据的 MasVnrType 为空而Area不为空，所以，填充方式按照类似poolQC和poolArea的方式，分组填充
# 其他数据中MasVnrType空值填fillna("None")，MasVnrArea空值填fillna(0)

#这部分数据很奇怪，类型为“None"类型，但是Area却大于0，还有等于1的
alldata[(alldata['MasVnrType']=='None')&(alldata['MasVnrArea']!=0)][['MasVnrType','MasVnrArea']]


alldata[alldata][['MasVnrArea']]['MasVnrArea'].value_counts()

#房子类型或者空间类型为空的话
print(alldata[alldata['MSSubClass'].isnull() | alldata['MSZoning'].isnull()][['MSSubClass','MSZoning']])
pd.crosstab(alldata.MSSubClass, alldata.MSZoning)
#通过观察30/70的subclass里 'RM'最多，20的subclass里'RL'最多。对应填充

#考虑到LotFrontage	与街道连接的线性脚与Neighborhood	房屋附近位置 存在一定的关系
 
print(alldata[["LotFrontage", "Neighborhood"]].isnull().sum())
print(alldata["LotFrontage"].value_counts().head(5)) 
 
# 考虑通过一定的方式来填充
# 例如：
alldata["LotFrontage"] = alldata.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

alldata["Neighborhood"].value_counts()

others = ['Functional','Utilities','SaleType','Electrical', "FireplaceQu",'Alley',"Fence", "MiscFeature",          'KitchenQual',"LotFrontage",'Exterior1st','Exterior2nd']
print(alldata[others].isnull().sum())
 
print(alldata['Functional'].value_counts().head(5)) # 填众数
print(alldata['Utilities'].value_counts().head(5)) # 填众数
print(alldata['SaleType'].value_counts().head(5)) # 填众数
print(alldata['Electrical'].value_counts().head(5)) # 填众数
print(alldata["Fence"].value_counts()) # 填众数
print(alldata["MiscFeature"].value_counts().head(5)) # 填众数
print(alldata['KitchenQual'].value_counts().head(5)) # 填众数
print(alldata['Exterior1st'].value_counts().head(5)) # 填众数
print(alldata['Exterior2nd'].value_counts().head(5)) # 填众数
print(alldata['FireplaceQu'].value_counts().head(5)) # 填'none'
print(alldata['Alley'].value_counts().head(5)) # 填'none'

#################################重复值处理###############################

alldata[alldata[alldata.columns].duplicated()==True]# 但是考虑到当前重复值后来不影响应用 所以可以不用删除

