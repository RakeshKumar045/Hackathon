import catboost as ctb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import seaborn as sns
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import shuffle

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# summarize the class distribution\n",
target = train.values[:, -1]
counter = Counter(target)
for k, v in counter.items():
    per = v / len(target) * 100
    print('Class=%d, Count=%d, Percentage=%.3f%%' % (k, v, per))

sns.heatmap(train.corr(), annot=True)

target = train['Surge_Pricing_Type']
del train['Surge_Pricing_Type']

data = pd.concat([train, test], axis=0).reset_index()

del data['index'], data['Trip_ID']

data['Confidence_Life_Style_Index'].fillna('Unknown', inplace=True)
data['Type_of_Cab'].fillna('Unknown', inplace=True)
data['Life_Style_Index'].fillna(data['Life_Style_Index'].mean(), inplace=True)
data['Customer_Since_Months'].fillna(-999, inplace=True)
data['Var1'].fillna(-999, inplace=True)

lbl = LabelEncoder()
a = lbl.fit_transform(data['Gender'])
a = a.reshape(-1, 1)
ohe = OneHotEncoder()
a = ohe.fit_transform(a).toarray()
a = a.astype(int)
a = pd.DataFrame(a, columns=['Female', 'Male'])
data = pd.concat([data, a], axis=1)
del data['Gender']

train_data = data[:train_size]
test_data = data[train_size:]

cat_var = [1, 4, 5]

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_data['target'] = target

train_data = shuffle(train_data)

target = train_data['target']

del train_data['target']

X_train, X_test, y_train, y_test = train_test_split(train_data, target, test_size=0.20)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20)

# grid = {'learning_rate': [0.01, 0.03, 0.1, 0.3],
#         'depth': [5, 6, 7],
#         'n_estimators':[2000, 3000, 4000],
#         'l2_leaf_reg': [1, 3, 5, 7, 9]}

del ctb_cf

ctb_cf = ctb.CatBoostClassifier(n_estimators=2500, l2_leaf_reg=5, depth=5, learning_rate=0.33,
                                eval_metric='Accuracy', cat_features=cat_var)

ctb_cf.fit(X_train, y_train, cat_features=cat_var, eval_set=(X_val, y_val), plot=True)

pred = ctb_cf.predict(X_test)

acc = accuracy_score(y_test, pred)

y_pred1 = ctb_cf.predict(test_data)

df = pd.DataFrame(test['Trip_ID'], columns=['Trip_ID'])

df['Surge_Pricing_Type'] = y_pred1

df.to_csv('ctb_prof.csv', sep=',', index=False)
