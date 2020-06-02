# In[1]
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.simplefilter('ignore')

import operator

import os

# In[2]
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
train['Loan_Amount_Requested'] = train['Loan_Amount_Requested'].str.replace(',', '').astype(float)

train.loc[train.Length_Employed == '< 1 year', 'Length_Employed'] = 0.5
train.loc[train.Length_Employed == '10+ years', 'Length_Employed'] = 10
train.loc[train.Length_Employed == '1 year', 'Length_Employed'] = 1
train.loc[train.Length_Employed == '2 years', 'Length_Employed'] = 2
train.loc[train.Length_Employed == '3 years', 'Length_Employed'] = 3
train.loc[train.Length_Employed == '4 years', 'Length_Employed'] = 4
train.loc[train.Length_Employed == '5 years', 'Length_Employed'] = 5
train.loc[train.Length_Employed == '6 years', 'Length_Employed'] = 6
train.loc[train.Length_Employed == '7 years', 'Length_Employed'] = 7
train.loc[train.Length_Employed == '8 years', 'Length_Employed'] = 8
train.loc[train.Length_Employed == '9 years', 'Length_Employed'] = 9
train.loc[train.Length_Employed == '', 'Length_Employed'] = -1

train.Length_Employed = train.Length_Employed.astype(float)

test['Loan_Amount_Requested'] = test['Loan_Amount_Requested'].str.replace(',', '').astype(float)

test.loc[test.Length_Employed == '< 1 year', 'Length_Employed'] = 0.5
test.loc[test.Length_Employed == '10+ years', 'Length_Employed'] = 10
test.loc[test.Length_Employed == '1 year', 'Length_Employed'] = 1
test.loc[test.Length_Employed == '2 years', 'Length_Employed'] = 2
test.loc[test.Length_Employed == '3 years', 'Length_Employed'] = 3
test.loc[test.Length_Employed == '4 years', 'Length_Employed'] = 4
test.loc[test.Length_Employed == '5 years', 'Length_Employed'] = 5
test.loc[test.Length_Employed == '6 years', 'Length_Employed'] = 6
test.loc[test.Length_Employed == '7 years', 'Length_Employed'] = 7
test.loc[test.Length_Employed == '8 years', 'Length_Employed'] = 8
test.loc[test.Length_Employed == '9 years', 'Length_Employed'] = 9
test.loc[test.Length_Employed == '', 'Length_Employed'] = -1

test.Length_Employed = test.Length_Employed.astype(float)

# In[3a]
train.loc[train.Home_Owner == 'None', 'Home_Owner'] = 'Other'
train.loc[pd.isnull(train['Home_Owner']), 'Home_Owner'] = 'Other'
# None, other have very low occurrence. Group with blank values.
train.loc[:, 'Annual_Income_missing'] = False
train.loc[pd.isnull(train['Annual_Income']), 'Annual_Income_missing'] = True

# train.loc[:, 'Months_Since_Deliquency_missing'] = False
# train.loc[pd.isnull(train['Months_Since_Deliquency']), 'Months_Since_Deliquency_missing'] = True
train.loc[pd.isnull(train['Months_Since_Deliquency']), 'Months_Since_Deliquency'] = -1

test.loc[test.Home_Owner == 'None', 'Home_Owner'] = 'Other'
test.loc[pd.isnull(test['Home_Owner']), 'Home_Owner'] = 'Other'
# None, other have very low occurrence. Group with blank values.
test.loc[:, 'Annual_Income_missing'] = False
test.loc[pd.isnull(test['Annual_Income']), 'Annual_Income_missing'] = True

test.loc[pd.isnull(test['Months_Since_Deliquency']), 'Months_Since_Deliquency'] = -1

# In[3b]
train['loan_income_ratio'] = train.Loan_Amount_Requested / train.Annual_Income
train['tot_debt_income_ratio'] = train.Debt_To_Income / 100 + train.Loan_Amount_Requested / train.Annual_Income
train['open_account_ratio'] = train.Number_Open_Accounts / train.Total_Accounts
train['income_emplength_ratio'] = train.Annual_Income / train.Length_Employed
train['loan_emplength_ratio'] = train.Loan_Amount_Requested / train.Length_Employed

test['loan_income_ratio'] = test.Loan_Amount_Requested / test.Annual_Income
test['tot_debt_income_ratio'] = test.Debt_To_Income / 100 + test.Loan_Amount_Requested / test.Annual_Income
test['open_account_ratio'] = test.Number_Open_Accounts / test.Total_Accounts
test['income_emplength_ratio'] = test.Annual_Income / test.Length_Employed
test['loan_emplength_ratio'] = test.Loan_Amount_Requested / test.Length_Employed

# In[3c]
train = pd.get_dummies(train, columns=['Home_Owner', 'Income_Verified', 'Purpose_Of_Loan', 'Gender'])
test = pd.get_dummies(test, columns=['Home_Owner', 'Income_Verified', 'Purpose_Of_Loan', 'Gender'])

# In[3d]
y = train.Interest_Rate - 1
train.drop(['Loan_ID', 'Interest_Rate'], inplace=True, axis=1)
test_ids = test.Loan_ID
test.drop(['Loan_ID'], inplace=True, axis=1)

# In[3e]
train.drop(['Length_Employed', 'Gender'], inplace=True, axis=1)
test.drop(['Length_Employed', 'Gender'], inplace=True, axis=1)


# In[3e1]
# for c in train.columns:
# if not c in test.columns:
# train.drop(c, axis=1, inplace=True)
def xgb_score(y_pred, dtrain):
    y_true = dtrain.get_label()
    # return 'f1_score', f1_score(y_true, np.round(y_pred))
    return 'f1_score', f1_score(y_true, y_pred, average='weighted')


# In[4]
params = {
    'eta': 0.008,  # use 0.002
    'max_depth': 8,
    'num_class': 3,
    'objective': 'multi:softmax',
    # 'eval_metric': 'error',
    'alpha': 10,
    'lambda': 6,
    'seed': 0,
    'silent': True
    # 'min_child_weight': 3,
    # 'subsample': 0.8,
    # 'scale_pos_weight': 1.5
}
# max_delta_step change 1-10 for unbalanced data
# scale_pos_weight - sum(negative cases) / sum(positive cases) for unbalanced data
# lambda, alpha, gamma - increase for conservative
# min_child_weight - default 1. inc for conservative. decrease for more fit
x1, x2, y1, y2 = train_test_split(train, y, test_size=0.2, random_state=0)
watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
model = xgb.train(params, xgb.DMatrix(x1, y1), 2000, watchlist, feval=xgb_score, maximize=True, verbose_eval=50,
                  early_stopping_rounds=1500)  # use 1500
# model = xgb.train(params, xgb.DMatrix(x1, y1), 2000, watchlist, maximize=False, verbose_eval=50, early_stopping_rounds=1500) #use 1500
# model = xgb.train(params, xgb.DMatrix(x1, y1), 2000, watchlist, maximize=True, verbose_eval=50, early_stopping_rounds=1500) #use 1500
# pred2 = np.round(model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit))
# print(sum(p2))
# p2[pred2==1] = 1
# print(sum(p2))
# np.mean(p2==y2)

# In[5]
# test.drop('subcatB00088', axis=1, inplace=True)
pred = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
pred = pred + 1

test_out = pd.DataFrame(test_ids)
test_out['Interest_Rate'] = pred.astype(int)
test_out.to_csv('pred4.csv', index=False)

# In[]
fscore = model.get_fscore()
sorted(fscore.items(), key=operator.itemgetter(1), reverse=True)
