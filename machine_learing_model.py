# coding: utf-8
import pandas as pd
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import  ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import itertools
import warnings
import xgboost as xgb
import csv

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

train = pd.read_csv("train_afterchange.csv")
test = pd.read_csv("test_afterchange.csv")
alldata = pd.concat((train.iloc[:, 1:-1], test.iloc[:, 1:]), ignore_index=True)
alldata.shape

X_train = train.iloc[:, 1:-1]
y = train.iloc[:, -1]
X_test = test.iloc[:, 1:]

# 定义验证函数
def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
    return (rmse)

#Lasso
clf1 = LassoCV(alphas=[1, 0.1, 0.001, 0.0005, 0.0003, 0.0002, 5e-4])
clf1.fit(X_train, y)
lasso_preds = np.expm1(clf1.predict(X_test))  # exp(x) - 1  <---->log1p(x)==log(1+x)
score1 = rmse_cv(clf1)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score1.mean(), score1.std()))

#ElasticNet
clf2 = ElasticNet(alpha=0.0005, l1_ratio=0.9)
clf2.fit(X_train, y)
elas_preds = np.expm1(clf2.predict(X_test))

score2 = rmse_cv(clf2)
print("\nElasticNet score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))

# print(lasso_preds)
# print(elas_preds)

# Id_list=[i for i in range(1461,2920)]
# print (len(Id_list))
# price_list=[]
# for i in range(0,1459):
#     new_list=[]
#     new_list=[Id_list[i],lasso_preds[i]]
#     price_list.append(new_list)
# print(price_list)
#
# e_price_list=[]
# for i in range(0,1459):
#     new_list=[]
#     new_list=[Id_list[i],elas_preds[i]]
#     e_price_list.append(new_list)
# print(e_price_list)

# fileHeader = ["Id", "SalePrice"]
# csvFile = open("sample_submission.csv", "w",newline='')
# writer = csv.writer(csvFile)
# writer.writerow(fileHeader)
# writer.writerows(e_price_list)
# csvFile.close()

#xgboost
clf3=xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0.045,
                 learning_rate=0.07,
                 max_depth=20,
                 min_child_weight=1.5,
                 n_estimators=300,
                 reg_alpha=0.65,
                 reg_lambda=0.45,
                 subsample=0.95)

clf3.fit(X_train, y.values)
xgb_preds = np.expm1(clf3.predict(X_test))


score3 = rmse_cv(clf3)
print("\nxgb score: {:.4f} ({:.4f})\n".format(score3.mean(), score3.std()))

final_result = 0.30 * lasso_preds + 0.50 * xgb_preds + 0.20 * elas_preds  # 0.11327

solution = pd.DataFrame({"id": test.index + 1461, "SalePrice": final_result}, columns=['id', 'SalePrice'])
solution.to_csv("sample_submission.csv", index=False)