# -*- coding: utf-8 -*-
"""
Created on Sat May 16 09:22:18 2020

@author: I354298
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

os.chdir(r"C:\Users\I354298\Desktop\SAP_laptop_backup\Desktop\AIML\Janatahack4")
data_train = pd.read_csv("train_Wc8LBpr.csv")
data_test = pd.read_csv("test_VsU9xXK.csv")

combine = pd.concat([data_train, data_test], ignore_index=True, sort=False)
null_dict = {}
for col in combine.columns:
    null_dict[col] = combine[col].isnull().sum()

custmon = []
for i in range(len(combine)):
    if combine["Customer_Since_Months"][i] == 10.0:
        custmon.append("Ten")
    elif combine["Customer_Since_Months"][i] >= 7.0:
        custmon.append("7to9")
    elif combine["Customer_Since_Months"][i] >= 4.0:
        custmon.append("4to6")
    elif combine["Customer_Since_Months"][i] >= 0.0:
        custmon.append("0to3")
    else:
        custmon.append(combine["Customer_Since_Months"][i])

combine["Customer_Since_Months"] = custmon

combine["Life_Style_Index"].fillna(np.mean(combine["Life_Style_Index"]), inplace=True)

combine["Var1"] = np.log(combine["Var1"])
combine["Var1"].fillna(np.mean(combine["Var1"]), inplace=True)
cols = list(combine.columns)

combine1 = combine.drop(columns=["Trip_ID", "Surge_Pricing_Type"])
cols1 = ["Customer_Since_Months", "Type_of_Cab", "Confidence_Life_Style_Index"]

for col in cols1:
    combine1[col] = combine1[col].fillna("Unknown")

cols = ["Type_of_Cab", "Customer_Since_Months", "Confidence_Life_Style_Index", "Destination_Type", "Gender"]
for col in cols:
    combine[col] = combine[col].astype("category")
combine1["Trip_ID"] = combine["Trip_ID"]
combine1["Surge_Pricing_Type"] = combine["Surge_Pricing_Type"]
combine1["Customer_Since_Months"].value_counts()
data_test = combine1[combine1["Surge_Pricing_Type"].isnull()]
data_test = data_test.drop(columns=["Surge_Pricing_Type", "Trip_ID"])
data_train = combine1[combine1["Surge_Pricing_Type"].notna()]
data_train["Surge_Pricing_Type"] = data_train["Surge_Pricing_Type"].astype("int")

# Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb

Y = data_train["Surge_Pricing_Type"]
X = data_train.drop(columns=["Surge_Pricing_Type", "Trip_ID"])
X1 = pd.get_dummies(X)
X_test = pd.get_dummies(data_test)
Y = Y - 1
evals_result = {}
feature_imp = pd.DataFrame()
features = [feat for feat in X1.columns]
folds = StratifiedKFold(n_splits=8, shuffle=False, random_state=8736)
param = {
    'bagging_freq': 125,
    'bagging_fraction': 0.9984231784564706,
    'boost_from_average': 'false',
    'boosting_type': 'gbdt',
    'feature_fraction': 0.54,
    'learning_rate': 0.005,
    'max_depth': -1,
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 16.0,
    'num_leaves': 40,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'multiclass',
    'num_class': 3,
    'verbosity': 1,
    "n_jobs": -1,
    "metric": "multi_logloss",
}

predictions = np.zeros((len(X1), 3))
predictions_test = np.zeros((len(X_test), 3))

for fold_, (train_idx, val_idx) in enumerate(folds.split(X1.values, Y.values)):
    print("Fold {}".format(fold_ + 1))
    # old_score = score
    d_train = lgb.Dataset(X1.iloc[train_idx][features], label=Y.iloc[train_idx])
    d_val = lgb.Dataset(X1.iloc[val_idx][features], label=Y.iloc[val_idx])
    num_round = 1000000
    clf = lgb.train(param, d_train, num_round, valid_sets=[d_train, d_val], verbose_eval=1000,
                    early_stopping_rounds=5000, evals_result=evals_result)
    oof = clf.predict(X1.iloc[val_idx][features], num_iteration=clf.best_iteration)
    # score = roc_auc_score(Y.iloc[val_idx],oof)
    fold_imp = pd.DataFrame()
    fold_imp["Feature"] = features
    fold_imp["importance"] = clf.feature_importance()
    fold_imp["fold"] = fold_ + 1
    feat_imp_df = pd.concat([feature_imp, fold_imp], axis=0)
    predictions += clf.predict(X1, num_iteration=clf.best_iteration)
    predictions_test += clf.predict(X_test, num_iteration=clf.best_iteration)
    # predictions = clf.predict(X_sub, num_iteration=clf.best_iteration)
    pred_lab = pd.DataFrame([np.argmax(pr) for pr in predictions])
    oof_lab = pd.DataFrame([np.argmax(pr) for pr in oof])
    acc_score = accuracy_score(Y, pred_lab)
    oof_acc = accuracy_score(Y.iloc[val_idx], oof_lab)
    print("OOF Accuracy {} and Training Accuracy {}".format(oof_acc, acc_score))

prediction_lab = pd.DataFrame([np.argmax(pr) for pr in predictions])
accuracy_score(Y, prediction_lab)
prediction_test_lab = pd.DataFrame([np.argmax(pr) for pr in predictions_test])
prediction_test_lab = prediction_test_lab + 1
test = list(combine1[combine1["Surge_Pricing_Type"].isnull()]["Trip_ID"])
sub = pd.DataFrame({"Trip_ID": test, "Surge_Pricing_Type": prediction_test_lab[0]})
sub.to_csv("predictions6.csv", index=False)

# XGBoost
from xgboost import XGBClassifier

predictions = np.zeros((len(X1), 3))
predictions_test1 = np.zeros((len(X_test), 3))
features = [feat for feat in X1.columns]
folds = StratifiedKFold(n_splits=8, shuffle=False, random_state=8736)
for fold_, (train_idx, val_idx) in enumerate(folds.split(X1, Y)):
    print("Fold {}".format(fold_ + 1))
    # old_score = score
    clf = XGBClassifier(n_estimators=800, verbosity=1, objective="multi:softprob", learning_rate=0.05, num_class=3,
                        eval_metric="mlogloss", early_stopping_rounds=10)
    clf.fit(X1.iloc[train_idx][features], Y.iloc[train_idx])
    best_iteration = clf.get_booster().best_ntree_limit
    oof = clf.predict_proba(X1.iloc[val_idx][features], ntree_limit=best_iteration)
    # score = roc_auc_score(Y.iloc[val_idx],oof)
    # fold_imp = pd.DataFrame()
    # fold_imp["Feature"] = features
    # fold_imp["importance"] = clf.feature_importance()
    # fold_imp["fold"] = fold_ +1
    # feat_imp_df = pd.concat([feature_imp,fold_imp], axis=0)
    predictions += clf.predict_proba(X1, ntree_limit=best_iteration)
    predictions_test1 += clf.predict_proba(X_test, ntree_limit=best_iteration)
    # predictions = clf.predict(X_sub, num_iteration=clf.best_iteration)
    pred_lab = pd.DataFrame([np.argmax(pr) for pr in predictions])
    oof_lab = pd.DataFrame([np.argmax(pr) for pr in oof])
    acc_score = accuracy_score(Y, pred_lab)
    oof_acc = accuracy_score(Y.iloc[val_idx], oof_lab)
    print("OOF accuracy {} and Training accuracy {}".format(oof_acc, acc_score))

prediction_lab = pd.DataFrame([np.argmax(pr) for pr in predictions])
accuracy_score(Y, prediction_lab)

prediction_test2 = predictions_test + predictions_test1

prediction_test_lab = pd.DataFrame([np.argmax(pr) for pr in prediction_test2])
prediction_test_lab = prediction_test_lab + 1
test = list(combine1[combine1["Surge_Pricing_Type"].isnull()]["Trip_ID"])
sub = pd.DataFrame({"Trip_ID": test, "Surge_Pricing_Type": prediction_test_lab[0]})
sub.to_csv("predictions7.csv", index=False)
