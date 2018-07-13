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


#####################################缺失值处理#####################################

##########对于pool的相关空值############
areamean = alldata.groupby('PoolQC')['PoolArea'].mean()
areamean

poolqcna = alldata[(alldata['PoolQC'].isnull())& (alldata['PoolArea']!=0)][['PoolQC','PoolArea']]
poolqcna


## 处理上述3个特殊情况的空值
for i in poolqcna.index:
    #定位到第i行的poolarea列，取其值
    v = alldata.loc[i,['PoolArea']].values
    print(type(np.abs(v-areamean)))
    #对第i行的poolqc列赋值：取和areamean差值最小的poolarea对应的poolqc
    alldata.loc[i,['PoolQC']] = np.abs(v-areamean).astype('float64').argmin()

    print(alldata.loc[i,['PoolQC']])
    
#给其他非异常空值分别赋值none和0
alldata['PoolQC'] = alldata["PoolQC"].fillna("None")
alldata['PoolArea'] = alldata["PoolArea"].fillna(0)

###########对于garage的相关空值############
alldata[['GarageCond','GarageFinish','GarageQual','GarageType']] = alldata[['GarageCond','GarageFinish','GarageQual','GarageType']].fillna('None')
alldata[['GarageCars','GarageArea']] = alldata[['GarageCars','GarageArea']].fillna(0)
alldata['Electrical'] = alldata['Electrical'].fillna( alldata['Electrical'].mode()[0])

###########对于basement的相关空值############
#找出所有前缀位bsmt的属性
a = pd.Series(alldata.columns)
BsmtList = a[a.str.contains('Bsmt')].values
 
condition = (alldata['BsmtExposure'].isnull()) & (alldata['BsmtCond'].notnull()) # 3个
#取众数（mode()[0]）填充
alldata.ix[(condition),'BsmtExposure'] = alldata['BsmtExposure'].mode()[0]
 
condition1 = (alldata['BsmtCond'].isnull()) & (alldata['BsmtExposure'].notnull()) # 3个
alldata.ix[(condition1),'BsmtCond'] = alldata.ix[(condition1),'BsmtQual']
 
condition2 = (alldata['BsmtQual'].isnull()) & (alldata['BsmtExposure'].notnull()) # 2个
alldata.ix[(condition2),'BsmtQual'] = alldata.ix[(condition2),'BsmtCond']
 
# 对于BsmtFinType1和BsmtFinType2
condition3 = (alldata['BsmtFinType1'].notnull()) & (alldata['BsmtFinType2'].isnull())
alldata.ix[condition3,'BsmtFinType2'] = 'Unf'
 
allBsmtNa = alldata_na.ix[BsmtList,:]
allBsmtNa_obj = allBsmtNa[allBsmtNa['dtype']=='object'].index
allBsmtNa_flo = allBsmtNa[allBsmtNa['dtype']!='object'].index
alldata[allBsmtNa_obj] =alldata[allBsmtNa_obj].fillna('None')
alldata[allBsmtNa_flo] = alldata[allBsmtNa_flo].fillna(0)

###############对MasVnr的相关空值：砖块类型和面积############
print(alldata[['MasVnrType', 'MasVnrArea']].isnull().sum())
 
print(len(alldata[(alldata['MasVnrType'].isnull())& (alldata['MasVnrArea'].isnull())])) # 23
 
print(len(alldata[(alldata['MasVnrType'].isnull())& (alldata['MasVnrArea'].notnull())]))
 
print(len(alldata[(alldata['MasVnrType'].notnull())& (alldata['MasVnrArea'].isnull())]))
 
print(alldata['MasVnrType'].value_counts())

# 有一条数据的 MasVnrType 为空而Area不为空，填充方式按照类似poolQC和poolArea的方式，用和area中位数最相近的数据组的type来填充
MasVnrM = alldata.groupby('MasVnrType')['MasVnrArea'].median()
print(MasVnrM)
mtypena = alldata[(alldata['MasVnrType'].isnull())& (alldata['MasVnrArea'].notnull())][['MasVnrType','MasVnrArea']]
print(mtypena)

for i in mastypena.index:
    v = alldata.loc[i,['MasVnrArea']].values
    alldata.loc[i,['MasVnrType']] = np.abs(v-MasVnrM).astype('float64').argmin()

# 其他数据中MasVnrType空值填fillna("None")，MasVnrArea空值填fillna(0)
alldata['MasVnrType'] = alldata["MasVnrType"].fillna("None")
alldata['MasVnrArea'] = alldata["MasVnrArea"].fillna(0)


###############对MS的相关空值:住处类型（层数年代等）和空间分类（经济适用、小区、商用等）##############
alldata["MSZoning"] = alldata.groupby("MSSubClass")["MSZoning"].transform(lambda x: x.fillna(x.mode()[0]))

##############对LorFrontage的相关空值##########
# 使用多项式拟合填充
 
x = alldata.loc[alldata["LotFrontage"].notnull(), "LotArea"]
y = alldata.loc[alldata["LotFrontage"].notnull(), "LotFrontage"]
t = (x <= 25000) & (y <= 150)
p = np.polyfit(x[t], y[t], 1)
alldata.loc[alldata['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, alldata.loc[alldata['LotFrontage'].isnull(), 'LotArea'])

###########处理其他有空值的feature################

alldata['KitchenQual'] = alldata['KitchenQual'].fillna(alldata['KitchenQual'].mode()[0]) # 用众数填充
alldata['Exterior1st'] = alldata['Exterior1st'].fillna(alldata['Exterior1st'].mode()[0])
alldata['Exterior2nd'] = alldata['Exterior2nd'].fillna(alldata['Exterior2nd'].mode()[0])
alldata["Functional"] = alldata["Functional"].fillna(alldata['Functional'].mode()[0])
alldata["SaleType"] = alldata["SaleType"].fillna(alldata['SaleType'].mode()[0])
alldata["Utilities"] = alldata["Utilities"].fillna(alldata['Utilities'].mode()[0])
 
alldata[["Fence", "MiscFeature"]] = alldata[["Fence", "MiscFeature"]].fillna('None')
alldata['FireplaceQu'] = alldata['FireplaceQu'].fillna('None')
alldata['Alley'] = alldata['Alley'].fillna('None')


## 年份虽然是数字，但不能使用众数填充的方式，故将年份按段映射进不同组里用YearGroupi的方式记录，再将空值填充为none
year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
#print(year_map)

# 将年份对应映射
alldata.GarageYrBlt = alldata.GarageYrBlt.map(year_map)
np.nan:False
#print(alldata['GarageYrBlt'])
alldata['GarageYrBlt']= alldata['GarageYrBlt'].fillna('None')# 必须 离散化之后再对应映射

#####################################异常值处理#####################################

#########由前面的数据分析发现房价和房子面积间存在异常值###########
plt.figure(figsize=(8,6))
plt.scatter(train.GrLivArea,train.SalePrice)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()

# 检查异常值GrLivArea>4000但是销售价格低于200000的记录
outliers_id = train[(train.GrLivArea>4000) & (train.SalePrice<200000)].index
print(outliers_id)


#########由前面的数据分析发现房价和地下室面积间存在异常值###########
plt.figure(figsize=(8,6))
plt.scatter(train.TotalBsmtSF,train.SalePrice)
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.show()

# 检查异常值TotalBsmtSF>4000但是销售价格低于200000的记录
outliers_id1 = train[(train.TotalBsmtSF>4000) & (train.SalePrice<200000)].index
print(outliers_id1)


alldata.to_csv('alldata_droppre.csv')


# 删除掉train里的上述异常值并重新绘制图片，发现两处的异常值都去掉了
train1=train.drop(outliers_id)
plt.figure(figsize=(8,6))
plt.scatter(train1.GrLivArea,train1.SalePrice)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.show()
plt.scatter(train1.TotalBsmtSF,train1.SalePrice)
plt.xlabel('TotalBsmtSF')
plt.ylabel('SalePrice')
plt.show()


# 在alldata里删除掉异常值（因前面分析需要，alldata里不含saleprice，删除并提取train里的saleprice列）
alldata=alldata.drop(outliers_id)
Y = train.SalePrice.drop(outliers_id)

# 将alldata的前半部分（处理了空值和异常值的traindata）截取出来，与train里处理过的saleprice结合写入新文件
train_now = pd.concat([alldata.iloc[:1458,:],Y], axis=1)
# 将alldata的后半部分（处理了空值的testdata）截取出来，写入新文件
test_now = alldata.iloc[1458:,:]
train_now.to_csv('Ctrain_afterclean.csv')
test_now.to_csv('test_afterclean.csv')

