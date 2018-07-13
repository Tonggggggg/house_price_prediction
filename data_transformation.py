# coding: utf-8

import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

# 读取清洗过的train和test data
train = pd.read_csv("train_afterclean.csv")
test = pd.read_csv("test_afterclean.csv")
alldata = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'], test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
alldata.shape


##################################处理有几种标准属性值的属性#################################

####### 1.处理属性值为Excellent(Ex)\Good(Gd)\AverageTypical(TA)\Fair(Fa)\Poor(Po)的属性 #######
# 处理序列型标称数据
ordinalList = ['ExterQual', 'ExterCond', 'GarageQual', 'GarageCond','PoolQC',              'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtQual','BsmtCond']
ordinalmap = {'Ex': 5,'Gd': 4,'TA': 3,'Fa': 2,'Po': 1,'None': 0}
for c in ordinalList:
    alldata[c] = alldata[c].map(ordinalmap) 


####### 2.处理属性值不固定的其他属性 #######
alldata['BsmtExposure'] = alldata['BsmtExposure'].map({'None':0, 'No':1, 'Mn':2, 'Av':3, 'Gd':4})    
alldata['BsmtFinType1'] = alldata['BsmtFinType1'].map({'None':0, 'Unf':1, 'LwQ':2,'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
alldata['BsmtFinType2'] = alldata['BsmtFinType2'].map({'None':0, 'Unf':1, 'LwQ':2,'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6})
alldata['Functional'] = alldata['Functional'].map({'Maj2':1, 'Sev':2, 'Min2':3, 'Min1':4, 'Maj1':5, 'Mod':6, 'Typ':7})
alldata['GarageFinish'] = alldata['GarageFinish'].map({'None':0, 'Unf':1, 'RFn':2, 'Fin':3})
alldata['Fence'] = alldata['Fence'].map({'MnWw':0, 'GdWo':1, 'MnPrv':2, 'GdPrv':3, 'None':4})
#print(alldata['GarageFinish'])

####### 3.处理属性值可以简单二分的（砌墙用的砖块类型 -> 用没用砖块；售房情况 -> 是否正常售出）######
MasVnrType_Any = alldata.MasVnrType.replace({'BrkCmn': 1,'BrkFace': 1,'CBlock': 1,'Stone': 1,'None': 0})
# MasVnrType_Any.name = 'MasVnrType_Any' #修改该series的列名
alldata.MasVnrType = MasVnrType_Any

alldata.rename(columns={'MasVnrType':'MasVnrType_Any'}, inplace=True) #修改该series的列名

SaleCondition_PriceDown = alldata.SaleCondition.replace({'Abnorml': 1,'Alloca': 1,'AdjLand': 1,'Family': 1,'Normal': 0,'Partial': 0})
# SaleCondition_PriceDown.name = 'SaleCondition_PriceDown' #修改该series的列名
alldata.SaleCondition = SaleCondition_PriceDown

alldata.rename(columns={'SaleCondition':'SaleCondition_PriceDown'}, inplace=True) #修改该series的列名


# 有没有中央空调
alldata = alldata.replace({'CentralAir': {'Y': 1,'N': 0}})
# 有没有车道
alldata = alldata.replace({'PavedDrive': {'Y': 1,'P': 0,'N': 0}})
# 建造年代晚于1946的住宅类型
newer_dwelling = alldata['MSSubClass'].map({20: 1,30: 0,40: 0,45: 0,50: 0,60: 1,70: 0,75: 0,80: 0,85: 0,90: 0,120: 1,150: 0,160: 1,180: 0,190: 0})

# newer_dwelling.name= 'newer_dwelling' #修改该series的列名
# alldata['MSSubClass'] = alldata['MSSubClass'].apply(str)
alldata.MSSubClass=newer_dwelling
alldata.rename(columns={'MSSubClass':'newer_dwelling'}, inplace=True) #修改该series的列名


# 周边地理位置比较好的，初始化一列全为0把几个属性值对应行赋值为1
Neighborhood_Good = pd.DataFrame(np.zeros((alldata.shape[0],1)), columns=['Neighborhood_Good'])
Neighborhood_Good[alldata.Neighborhood=='NridgHt'] = 1
Neighborhood_Good[alldata.Neighborhood=='Crawfor'] = 1
Neighborhood_Good[alldata.Neighborhood=='StoneBr'] = 1
Neighborhood_Good[alldata.Neighborhood=='Somerst'] = 1
Neighborhood_Good[alldata.Neighborhood=='NoRidge'] = 1
alldata.Neighborhood = Neighborhood_Good

# Neighborhood_Good.name='Neighborhood_Good'# 修改该列列名
alldata.rename(columns={'Neighborhood':'Neighborhood_Good'}, inplace=True) #修改该series的列名


# 售出月份在5，6,7月的为1（量比较大）

# season = (alldata['MoSold'].isin([5,6,7]))*1 #(@@@@@)
# season.name='season'
# alldata['MoSold'] = alldata['MoSold'].apply(str)

season = pd.DataFrame(np.zeros((alldata.shape[0],1)), columns=['season'])
season[alldata.MoSold==5] = 1
season[alldata.MoSold==6] = 1
season[alldata.MoSold==7] = 1
alldata.MoSold = season
alldata.rename(columns={'MoSold':'season'}, inplace=True) #修改该series的列名



##################################### 4.处理质量相关属性#####################################
############# 总体材质评分（1-10）总体状态评分（1-10）外部材质评分（1-5）外部状态评分（1-5）##################
############# 地下室状态评分（1-5）车库质量评分（1-5）车库状态评分（1-5）厨房质量评分（1-5）##################


# 处理OverallQual：将该属性分成两个子属性，以5为分界线。大于5及小于5的再分别归入新列
overall_poor_qu = alldata.OverallQual.copy()# Series类型
overall_poor_qu = 5 - overall_poor_qu #-5 —— 4
overall_poor_qu[overall_poor_qu<0] = 0 # -5，-4，-3，-2，-1设为0，得出0-4的数列
overall_poor_qu.name = 'overall_poor_qu'
overall_good_qu = alldata.OverallQual.copy()
overall_good_qu = overall_good_qu - 5 #-4 ——5
overall_good_qu[overall_good_qu<0] = 0 #-4，-3，-2，-1设为0，得出0-5的数列
overall_good_qu.name = 'overall_good_qu'
 
# 处理OverallCond ：将该属性分成两个子属性，以5为分界线，大于5及小于5的再分别归入新列
overall_poor_cond = alldata.OverallCond.copy()# Series类型
overall_poor_cond = 5 - overall_poor_cond
overall_poor_cond[overall_poor_cond<0] = 0
overall_poor_cond.name = 'overall_poor_cond'
overall_good_cond = alldata.OverallCond.copy()
overall_good_cond = overall_good_cond - 5
overall_good_cond[overall_good_cond<0] = 0
overall_good_cond.name = 'overall_good_cond'
 
# 处理ExterQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别归入新列
exter_poor_qu = alldata.ExterQual.copy()
exter_poor_qu[exter_poor_qu<3] = 1
exter_poor_qu[exter_poor_qu>=3] = 0
exter_poor_qu.name = 'exter_poor_qu'
exter_good_qu = alldata.ExterQual.copy()
exter_good_qu[exter_good_qu<=3] = 0
exter_good_qu[exter_good_qu>3] = 1
exter_good_qu.name = 'exter_good_qu'
 
# 处理ExterCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别归入新列
exter_poor_cond = alldata.ExterCond.copy()
exter_poor_cond[exter_poor_cond<3] = 1
exter_poor_cond[exter_poor_cond>=3] = 0
exter_poor_cond.name = 'exter_poor_cond'
exter_good_cond = alldata.ExterCond.copy()
exter_good_cond[exter_good_cond<=3] = 0
exter_good_cond[exter_good_cond>3] = 1
exter_good_cond.name = 'exter_good_cond'
 
# 处理BsmtCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别归入新列
bsmt_poor_cond = alldata.BsmtCond.copy()
bsmt_poor_cond[bsmt_poor_cond<3] = 1
bsmt_poor_cond[bsmt_poor_cond>=3] = 0
bsmt_poor_cond.name = 'bsmt_poor_cond'
bsmt_good_cond = alldata.BsmtCond.copy()
bsmt_good_cond[bsmt_good_cond<=3] = 0
bsmt_good_cond[bsmt_good_cond>3] = 1
bsmt_good_cond.name = 'bsmt_good_cond'
 
# 处理GarageQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别归入新列
garage_poor_qu = alldata.GarageQual.copy()
garage_poor_qu[garage_poor_qu<3] = 1
garage_poor_qu[garage_poor_qu>=3] = 0
garage_poor_qu.name = 'garage_poor_qu'
garage_good_qu = alldata.GarageQual.copy()
garage_good_qu[garage_good_qu<=3] = 0
garage_good_qu[garage_good_qu>3] = 1
garage_good_qu.name = 'garage_good_qu'
 
# 处理GarageCond：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别归入新列
garage_poor_cond = alldata.GarageCond.copy()
garage_poor_cond[garage_poor_cond<3] = 1
garage_poor_cond[garage_poor_cond>=3] = 0
garage_poor_cond.name = 'garage_poor_cond'
garage_good_cond = alldata.GarageCond.copy()
garage_good_cond[garage_good_cond<=3] = 0
garage_good_cond[garage_good_cond>3] = 1
garage_good_cond.name = 'garage_good_cond'
 
# 处理KitchenQual：将该属性分成两个子属性，以3为分界线，大于3及小于3的再分别归入新列
kitchen_poor_qu = alldata.KitchenQual.copy()
kitchen_poor_qu[kitchen_poor_qu<3] = 1
kitchen_poor_qu[kitchen_poor_qu>=3] = 0
kitchen_poor_qu.name = 'kitchen_poor_qu'
kitchen_good_qu = alldata.KitchenQual.copy()
kitchen_good_qu[kitchen_good_qu<=3] = 0
kitchen_good_qu[kitchen_good_qu>3] = 1
kitchen_good_qu.name = 'kitchen_good_qu'

# 将构造的属性合并
qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
                     exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,
                     garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)


#################################### 5.处理时间相关属性 ###################################

# 不一致的建造年份与改建年份
Xremoded = (alldata['YearBuilt']!=alldata['YearRemodAdd'])*1 #(@@@@@)
# 改建年份晚于售出年份
Xrecentremoded = (alldata['YearRemodAdd']>=alldata['YrSold'])*1 #(@@@@@)
# 建造年份晚于售出年份
XnewHouse = (alldata['YearBuilt']>=alldata['YrSold'])*1 #(@@@@@)
#计算2010年止的房屋年龄，2010年止售出了多少年，改建到售出花了多少年
XHouseAge = 2010 - alldata['YearBuilt']
XTimeSinceSold = 2010 - alldata['YrSold']
XYearSinceRemodel = alldata['YrSold'] - alldata['YearRemodAdd']
 
Xremoded.name='Xremoded'
Xrecentremoded.name='Xrecentremoded'
XnewHouse.name='XnewHouse'
XTimeSinceSold.name='XTimeSinceSold'
XYearSinceRemodel.name='XYearSinceRemodel'
XHouseAge.name='XHouseAge'
 
#将构造的新属性合并
year_list = pd.concat((Xremoded,Xrecentremoded,XnewHouse,XHouseAge,XTimeSinceSold,XYearSinceRemodel),axis=1)

#将其他的时间相关属性离散化，使年份对应映射
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
# alldata.GarageYrBlt = alldata.GarageYrBlt.map(year_map) # 在数据填充时已经完成该转换了
alldata.YearRemodAdd = alldata.YearRemodAdd.map(year_map)


#################################### 6.构造price_category新属性 ###################################


from sklearn.svm import SVC
svm = SVC(C=100, gamma=0.0001, kernel='rbf')
 
pc = pd.Series(np.zeros(train.shape[0]))
pc[:] = 'pc1'
pc[train.SalePrice >= 150000] = 'pc2'
pc[train.SalePrice >= 220000] = 'pc3'


columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
X_t = pd.get_dummies(train.loc[:, columns_for_pc], sparse=True)
svm.fit(X_t, pc)# 训练
p = train.SalePrice/100000
print(p)

price_category = pd.DataFrame(np.zeros((alldata.shape[0],1)), columns=['pc'])
X_t = pd.get_dummies(alldata.loc[:, columns_for_pc], sparse=True)
pc_pred = svm.predict(X_t) # 预测
 
price_category[pc_pred=='pc2'] = 1
price_category[pc_pred=='pc3'] = 2

price_category.name='price_category'

print(price_category['pc'].value_counts())


# 将数字类型的属性用quantile取四分之三位的值，将所有数据按照比例缩放
numeric_feats = alldata.dtypes[alldata.dtypes != "object"].index
t = alldata[numeric_feats].quantile(.75) # 取四分之三分位
use_75_scater = t[t != 0].index #print(use_75_scater)
alldata[use_75_scater] = alldata[use_75_scater]/alldata[use_75_scater].quantile(.75)
#print(alldata[use_75_scater])


# 标准化数据使符合正态分布
from scipy.special import boxcox1p
 
t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
     '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

#log1p -> log(1+x)
train["SalePrice"] = np.log1p(train["SalePrice"]) # 对于SalePrice 采用log1p较好---np.expm1(clf1.predict(X_test))
print(train["SalePrice"])
lam = 0.15 # 100 * (1-lam)% confidence
for feat in t:
    alldata[feat] = boxcox1p(alldata[feat], lam)  # 对于其他属性，采用boxcox1p较好
# 将标称型变量用【get_dummies】函数二值化
X = pd.get_dummies(alldata)
print(alldata.shape)
print(X.shape)
X = X.fillna(X.mean())

#输出每列每种值的出现次数发现有几个出现非常少
# object_feats = alldata.dtypes[alldata.dtypes == "object"].index
# T = alldata[object_feats]
# for i in range(0,len(T)):
#     print(T.iloc[:,i].value_counts())

# 把他们去掉，将处理好和添加的新属性们合成一个新数据集
# X = X.drop('Condition2_PosN', axis=1)
# X = X.drop('MSZoning_C (all)', axis=1)
# X = X.drop('MSSubClass_160', axis=1)
X= pd.concat((X, newer_dwelling, season, year_list ,qu_list,MasVnrType_Any,price_category,SaleCondition_PriceDown,Neighborhood_Good), axis=1)
print(X.shape)


from itertools import product, chain

#定义一个能够在指定的属性间创建新属性的函数
def poly(X):
    #areas是5个和面积相关的属性
    areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']
    #lists是前面自行集合的几个列表的属性名（好坏质量集合，年代集合，总质量集合）
    lists = chain(qu_list.axes[1].get_values(),year_list.axes[1].get_values(),ordinalList,
                  ['MasVnrType_Any'])  #,'Neighborhood_Good','SaleCondition_PriceDown'

    #【product】函数为areas到lists一个个的映射（eg:LotArea -> overall_poor_qu）
    for a, t in product(areas, lists):
        # print(a,t)
        # 返回每行X中a列和t列的乘积【prod】,且将新属性命名为LotArea_overall_poor_qu等
        x = X.loc[:, [a, t]].prod(1) 
        x.name = a + '_' + t
        # print(x)
        # break
        yield x  # yield不执行任何函数代码，每执行到一个 yield 语句就会中断，并返回一个迭代值，但不打断for循环


XP = pd.concat(poly(X), axis=1)
print(XP.shape)
X = pd.concat((X, XP), axis=1)
print(X.shape)
X_train = X[:train.shape[0]]
X_test = X[train.shape[0]:]

print(X_train.shape)

Y= train.SalePrice
train_now = pd.concat([X_train,Y], axis=1)
test_now = X_test
print(test_now.shape)


train_now.to_csv('train_afterchange.csv')
test_now.to_csv('test_afterchange.csv')

test = 6;
test1= np.log1p(test)
test2 = np.expm1(test1)
print(test2)

