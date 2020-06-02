import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.isnull().sum()
# converting variables to  numeric
train_1 = train[train.Interest_Rate == 1]
train_2 = train[train.Interest_Rate == 2]
train_3 = train[train.Interest_Rate == 3]

train_under = pd.concat([train_1, train_2, train_3], axis=0)

df = pd.concat([train_under, test], axis=0)

df.isnull().sum()

train.Interest_Rate.value_counts()

# employee lenght

df['Length_Employed'].replace({'< 1 year': '0', '1 year': '1', '10+ years': '10'}, inplace=True)
df['Length_Employed'] = df['Length_Employed'].str.replace('years', '')

df['Length_Employed'].fillna(-1, inplace=True)
df['Length_Employed'] = df['Length_Employed'].astype(int)

df['Length_Employed'] = pd.cut(df['Length_Employed'], bins=[-2, -1, 1, 3, 6, 9, 11], labels=[-1, 1, 2, 3, 4, 10])

df['Length_Employed'] = df['Length_Employed'].astype(int)
df['Length_Employed'].value_counts()

# HOme owned

sns.countplot(df.Home_Owner)
df.Home_Owner = df.Home_Owner.map(lambda x: 3 if x == 'Own' else 2 if x == 'Mortgage'
else 1 if x == 'Rent' else -1)
df.Home_Owner.value_counts()
# loan amount
df.Loan_Amount_Requested = df.Loan_Amount_Requested.str.replace(',', '').astype(int)

df.Loan_Amount_Requested = np.cbrt(df.Loan_Amount_Requested)

sns.boxplot(df.Interest_Rate, df.Loan_Amount_Requested)

sns.distplot(df.Loan_Amount_Requested)

# =============================================================================
# df.Loan_Amount_Requested = pd.cut(df.Loan_Amount_Requested,  bins = [500,4000,6000, 8000,10000,
#                                                           12000,16000,20000,25000,30000, 36000],
#                       labels = [1,2,3,4,5,6,7,8,9,10])
# sns.distplot(a)
# df.Loan_Amount_Requested.describe()
# =============================================================================

# income verified

sns.countplot(df.Income_Verified)
df.Income_Verified = df.Income_Verified.map(lambda x: 1 if x == 'not verified'
else 3 if x == 'VERIFIED - income' else 2)

# Purpose

pd.crosstab(df.Purpose_Of_Loan, df.Interest_Rate).plot.bar()

df.Purpose_Of_Loan = df.Purpose_Of_Loan.map(lambda x: 3 if x == 'debt_consolidation'
else 2 if x == 'credit_card' else -1)
df.Purpose_Of_Loan.astype(int)

# debt

df.Debt_To_Income.describe()

sns.distplot(df.Debt_To_Income)
# =============================================================================
# df.Debt_To_Income.isnull().sum()
# 
# df['Debt_To_Income'] = pd.cut(df.Debt_To_Income, bins = [-1,4,8,12,16,20,24,28,32,41],
#                      labels = [1,2,3,4,5,6,7,8,9])
# =============================================================================

# Inquiries_Last_6Mo
sns.countplot(df.Inquiries_Last_6Mo)
df.Inquiries_Last_6Mo.value_counts()

df.Inquiries_Last_6Mo = df.Inquiries_Last_6Mo.map(lambda x: 3 if x >= 3 else x)

# Months_Since_Deliquency

sns.distplot(df.Months_Since_Deliquency)

df.Months_Since_Deliquency.describe()


def IQR(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    upper = df[variable].quantile(0.75) + (IQR * distance)
    lower = df[variable].quantile(0.25) - (IQR * distance)
    return lower, upper


IQR(df, 'Months_Since_Deliquency', 1.5)

df.Months_Since_Deliquency = df.Months_Since_Deliquency.map(lambda x: 95 if x > 95 else x)
sns.distplot(df.Months_Since_Deliquency)

df.Months_Since_Deliquency.fillna(0, inplace=True)

df.Months_Since_Deliquency = np.cbrt(df.Months_Since_Deliquency)
df.Months_Since_Deliquency.isna().sum()

df.Months_Since_Deliquency.describe()

# =============================================================================
#     
# df.Months_Since_Deliquency = pd.cut(df.Months_Since_Deliquency,
#                           bins = [-1,0,20,40,60,8050,200],
#                           labels = [0,1,2,3])
# =============================================================================
sns.countplot(df.Months_Since_Deliquency)

pd.crosstab(df.Months_Since_Deliquency, df.Interest_Rate)
# Total_Accounts
sns.distplot(df.Total_Accounts)
df.Total_Accounts.describe()

IQR(df, 'Total_Accounts', 1.5)

df.Total_Accounts = df.Total_Accounts.map(lambda x: 58 if x > 58 else x)

df.Total_Accounts = np.sqrt(df.Total_Accounts)

# df.Total_Accounts = pd.cut(df.Total_Accounts, bins = [0,8,12,18,24,30,36,42,48,160],
#  labels = [1,2,3,4,5,6,7,8,9])
df.Total_Accounts = df.Total_Accounts.astype(int)
# Number_Open_Accounts

sns.distplot(df.Number_Open_Accounts)

IQR(df, 'Number_Open_Accounts', 1.5)
import math

df.Number_Open_Accounts = df.Number_Open_Accounts.map(lambda x: 25 if x > 25 else x)

# df.Number_Open_Accounts =  pd.cut(df.Number_Open_Accounts, bins = [-1,4,6,8,10,12,14,16,18,20,86],
#     labels = [1,2,3,4,5,6,7,8,9,10])
df.Number_Open_Accounts = df.Number_Open_Accounts.astype(int)

df.Number_Open_Accounts = np.sqrt(df.Number_Open_Accounts)
df.Number_Open_Accounts.value_counts()

a = np.cbrt(df.Number_Open_Accounts)
sns.distplot(a)

# Closed account

df['closed_accounts'] = df['Total_Accounts'] - df['Number_Open_Accounts']

IQR(df, 'closed_accounts', 1.5)

df.closed_accounts.describe()

sns.distplot(df.closed_accounts)

# gender 

df.Gender.replace({'Female': 1, 'Male': 0}, inplace=True)

# Annual_Income

df.Annual_Income.describe().apply(lambda x: format(x, 'f'))
df.Annual_Income.median()
IQR(df, 'Annual_Income', 1.5)

df.Annual_Income.describe()

df.Annual_Income = df.Annual_Income.map(lambda x: 155000 if x > 155000 else x)
# df.Annual_Income.fillna(df.Annual_Income.median(), inplace = True)


# df.Annual_Income.fillna(df.Annual_Income.mean(), inplace = True)

# df['ann'] = df.Annual_Income
# =============================================================================
# df['ann'] = df.Annual_Income
# 
df.Annual_Income = df.groupby('Total_Accounts')['Annual_Income'].transform(lambda x: x.fillna(x.mean()))
df.Annual_Income = np.cbrt(df.Annual_Income)

# df.Annual_Income = df.Annual_Income.astype(int)
# df.ann.describe()
# df.ann.isna().sum()
# =============================================================================


df.isna().sum()
# df.Annual_Income = df.groupby('Debt_To_Income')['Annual_Income'].transform(lambda x: x.fillna(x.mean()))
sns.distplot(df.Annual_Income)

# =============================================================================
# a= np.cbrt(df.Annual_Income)
# sns.distplot(a)
# 
# =============================================================================
df.Annual_Income.describe()

# =============================================================================
# # df.Annual_Income =  pd.cut(df.Annual_Income, bins = [0,10000,20000,30000,40000,50000,60000,70000,80000,90000,
# #                                       100000,110000,120000,130000,140000,150000,160000],
# #                      labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
# # 
# # sns.distplot(a)
# 
# =============================================================================
corr = df.corr(method='spearman')
df.isnull().sum()

df = df.drop(['Loan_ID', 'Gender'], axis=1)

# model
training = df.iloc[:len(train_under), :]
testing = df.iloc[len(train_under):, :]
testing = testing.drop('Interest_Rate', axis=1)

# Model Creation

X = training.drop('Interest_Rate', axis=1)
y = training['Interest_Rate']

# =============================================================================
# from imblearn.over_sampling import SMOTE
# 
# sm = SMOTE(random_state = 42, k_neighbors = 5, n_jobs = -1)
# 
# X_res, y_res = sm.fit_sample(X, y)
# =============================================================================
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.18, random_state=21)

from sklearn.metrics import accuracy_score, classification_report, f1_score
from lightgbm import LGBMClassifier

class_1 = (len(y) - len(y[y == 1])) / len(y)
class_2 = (len(y) - len(y[y == 2])) / len(y)
class_3 = (len(y) - len(y[y == 3])) / len(y)

weight = {1: '0.76', 2: '0.6', 3: '0.63'}

lgb = LGBMClassifier(boosting_type='gbdt',
                     max_depth=6,
                     learning_rate=0.02,
                     n_estimators=5000,
                     class_weight=weight,
                     min_child_weight=0.001,
                     colsample_bytree=0.5,
                     reg_lambda=0.0001,
                     random_state=7,
                     objective='multiclass')

lgb.fit(x_train, y_train,
        eval_set=[(x_train, y_train), (x_val, y_val.values)],
        early_stopping_rounds=500,
        verbose=200)

lgb.fit(X, y)

print(lgb.feature_importances_)

plt.bar(X.columns, lgb.feature_importances_)
plt.xticks(rotation=90)

print(accuracy_score(y_val, lgb.predict(x_val)))

submission = pd.DataFrame()
submission['Loan_ID'] = test['Loan_ID']
submission['Interest_Rate'] = lgb.predict(testing)
submission.to_csv('weights.csv', index=False, header=True)

# kfold
# =============================================================================
# from sklearn.model_selection import StratifiedKFold
# 
# err = []
# y_pred_tot_lgm = []
# 
# from sklearn.model_selection import StratifiedKFold
# 
# fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=2020)
# i = 1
# for train_index, test_index in fold.split(X_res, y_res):
#     x_train, x_val = X_res.iloc[train_index], X_res.iloc[test_index]
#     y_train, y_val = y_res.iloc[train_index], y_res.iloc[test_index]
#     m = LGBMClassifier(boosting_type='gbdt',
#                        max_depth=7,
#                        learning_rate=0.05,
#                        n_estimators=5000,
#                        colsample_bytree=0.7,
#                        random_state=1994,
#                        objective='multiclass')
#     m.fit(x_train, y_train,
#           eval_set=[(x_train,y_train),(x_val, y_val)],
#           early_stopping_rounds=200,
#           verbose=200)
#     pred_y = m.predict(x_val)
#     print(i, " err_lgm: ", accuracy_score(y_val, pred_y))
#     err.append(accuracy_score(y_val, pred_y))
#     pred_test = m.predict(testing)
#     i = i + 1
#     y_pred_tot_lgm.append(pred_test)
#     
# np.mean(err,0)
# 
# err[6]
# 
# submission = pd.DataFrame()
# submission['Loan_ID'] = test['Loan_ID']
# submission['Interest_Rate'] = y_pred_tot_lgm[6]
# submission.to_csv('normal_kfolds.csv', index=False, header=True)
# =============================================================================
