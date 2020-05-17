# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:17:22 2020

@author: admin
"""

import pandas as pd

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.cbook import boxplot_stats

% matplotlib
inline

import numpy as np
import scipy.stats as ss
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

train = pd.read_csv('D:/Data Analytics - Personal/Analytics Vidhya/Cab Service/train_Wc8LBpr.csv', index_col='Trip_ID')
test = pd.read_csv('D:/Data Analytics - Personal/Analytics Vidhya/Cab Service/test_VsU9xXK.csv', index_col='Trip_ID')


def desc_data(df, pred="None"):
    obs = df.shape[0]
    types = df.dtypes
    counts = df.apply(lambda x: x.count())
    uniques = df.apply(lambda x: [x.unique()])
    nulls = df.apply(lambda x: x.isnull().sum())
    distincts = df.apply(lambda x: x.unique().shape[0])
    missing_ration = (df.isnull().sum() / obs) * 100
    skewness = df.skew()
    kurtosis = df.kurt()
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing ration', 'skewness', 'kurtosis']
        str = pd.concat([types, counts, distincts, nulls, missing_ration, skewness, kurtosis], axis=1)

    else:
        corr = df.corr()[pred]
        str = pd.concat([types, counts, distincts, nulls, missing_ration, skewness, kurtosis, corr], axis=1, sort=False)
        corr_col = 'corr ' + pred
        cols = ['types', 'counts', 'distincts', 'nulls', 'missing_ration', 'skewness', 'kurtosis', corr_col]

    str.columns = cols
    dtypes = str.types.value_counts()
    print('___________________________\nData types:\n', str.types.value_counts())
    print('___________________________')
    return str


details = desc_data(train, 'Surge_Pricing_Type')

display(train.describe().transpose())


###########################################################################

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


obj_col = [col for col in train.columns if train[col].dtypes == "object"]

for col in obj_col:
    col_target = pd.crosstab(train[col], train['Surge_Pricing_Type']).values
    print(' corr with ', col, ' - ', cramers_v(col_target))

#  corr with  Type_of_Cab  -  0.5847978171536695
#  corr with  Confidence_Life_Style_Index  -  0.15709294188993114
#  corr with  Destination_Type  -  0.12910616612106834
#  corr with  Gender  -  0.0017291068754115273

################################## Data Handling ##########################

var1_median = train.Var1.dropna().median()
var1_median

var2_median = train.Var2.dropna().median()
var2_median

var3_median = train.Var3.dropna().median()
var3_median

cust_months_median = train.Customer_Since_Months.dropna().median()
cust_months_median

lif_index_median = train.Life_Style_Index.dropna().median()
lif_index_median

combine = [train, test]

for dataset in combine:
    dataset['Var1'] = dataset['Var1'].fillna(var1_median)
    dataset['Var2'] = dataset['Var2'].fillna(var2_median)
    dataset['Var3'] = dataset['Var3'].fillna(var3_median)
    dataset['Customer_Since_Months'] = dataset['Customer_Since_Months'].fillna(cust_months_median)
    dataset['Life_Style_Index'] = dataset['Life_Style_Index'].fillna(lif_index_median)
    dataset['Confidence_Life_Style_Index'] = dataset['Confidence_Life_Style_Index'].fillna('None')
#   
combine = [train, test]
for dataset in combine:
    dataset['Type_of_Cab'] = dataset['Type_of_Cab'].fillna('None')

###############################################################################


################################## Data Visualization ##########################

fig = plt.figure(figsize=(20, 15))
sns.set(font_scale=1.5)

fig2 = fig.add_subplot(222);
sns.scatterplot(x=train.Var1, y=train.Surge_Pricing_Type, palette='Spectral')

fig3 = fig.add_subplot(223);
sns.scatterplot(x=train.Var2, y=train.Surge_Pricing_Type, palette='Spectral')

fig4 = fig.add_subplot(224);
sns.scatterplot(x=train.Var3, y=train.Surge_Pricing_Type, palette='Spectral')

####################################  Deleting Outliers ####################################

train = train.drop(train[(train.Var1 >= 200)].index)

train = train.drop(train[(train.Var2 >= 100)].index)

train = train.drop(train[(train.Var3 >= 150)].index)
train.shape

###################################
train = train.drop(train[(train.Life_Style_Index > 4.5)].index)
train.shape

###################################
train = train.drop(train[(train.Customer_Since_Months == 0) & (train.Cancellation_Last_1Month > 0)].index)
train.shape

###################################
train = train.drop(train[(train['Surge_Pricing_Type'] == 1) & (train['Customer_Rating'] <= .4793)].index)
train.shape

train = train.drop(train[(train['Surge_Pricing_Type'] == 2) & (train['Customer_Rating'] <= .0562)].index)
train.shape

###################################
train = train.drop(train[(train.Life_Style_Index > 4) & (train.Confidence_Life_Style_Index == 'A')].index)
train.shape

train = train.drop(train[(train.Life_Style_Index > 4) & (train.Confidence_Life_Style_Index == 'B')].index)
train.shape

train = train.drop(train[(train.Life_Style_Index > 4) & (train.Confidence_Life_Style_Index == 'C')].index)
train.shape

###################################


###################################################################################


###################################################################################

sns.boxplot(x='Type_of_Cab', y='Life_Style_Index', data=train[['Type_of_Cab', 'Life_Style_Index']])

df = train[train['Destination_Type'] == 'F']['Customer_Rating']

q1 = np.percentile(df, 25)
q3 = np.percentile(df, 75)
iqr = q3 - q1

# max_tr = q3 + (1.5*iqr)
min_tr = q1 - (1.5 * iqr)
min_tr
del df

df_new = train[(train['Life_Style_Index'] < 1 'F') & (train['Customer_Rating'] <= 0.3421)]
df_new.shape
del df_new
# out_rating = boxplot_stats(train.Customer_Rating).pop(0)['fliers']


fig5 = fig.add_subplot(222);
sns.scatterplot(x=train.Customer_Rating, y=train.Type_of_Cab, palette='Spectral')

df = train[(train.Life_Style_Index < 1.75)]
df.shape

# plt.figure(figsize=(20,6))
# g = sns.FacetGrid(train , col = "Surge_Pricing_Type")
# g.map(plt.hist, "Gender" , bins=20)

# train[['Trip_Distance', 'Surge_Pricing_Type']].groupby(['Trip_Distance'], as_index=False).mean().sort_values(by='Surge_Pricing_Type', ascending=False)


################################### Modelling ##################################

y = train.Surge_Pricing_Type
train.drop(['Surge_Pricing_Type'], axis=1, inplace=True)

numerical_col = [col for col in train.columns if train[col].dtypes in ["int64", "float64", "int32"]]

object_cols = [col for col in train.columns if train[col].dtypes == "object"]

object_nunique = list(map(lambda col: train[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

sorted(d.items())

low_cardinality_cols = [col for col in object_cols if train[col].nunique() < 15]

x_train, x_valid, y_train, y_valid = train_test_split(train, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_col),
        ('cat', categorical_transformer, low_cardinality_cols)
    ])

data_full = pd.concat([x_train, x_valid, test])[numerical_col + low_cardinality_cols]

preprocessor.fit(data_full)

x_train_full = preprocessor.transform(x_train[numerical_col + low_cardinality_cols])
x_valid_full = preprocessor.transform(x_valid[numerical_col + low_cardinality_cols])
test_full = preprocessor.transform(test[numerical_col + low_cardinality_cols])

######################################## XGBOOST ###################################


xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05)
xgb_model.fit(x_train_full, y_train, early_stopping_rounds=5, eval_set=[(x_valid_full, y_valid)])
xgb_pred = xgb_model.predict(x_valid_full)

# xgb_roc = roc_auc_score(y_valid, xgb_pred)
print("accuracy xgb: ", accuracy_score(y_valid, xgb_pred))

####################################################################################

rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(x_train_full, y_train)
rf_pred = rf_model.predict(x_valid_full)

# rf_roc = roc_auc_score(y_valid, rf_pred)
print("accuracy rf: ", accuracy_score(y_valid, rf_pred))

###################################################################################

lgbm_model = LGBMClassifier(max_depth=-1,
                            n_estimators=500,
                            learning_rate=0.1,
                            colsample_bytree=.2,
                            objective='multiclass',
                            n_jobs=-1)

lgbm_model.fit(x_train_full, y_train, eval_metric='multi_logloss', eval_set=[(x_valid_full, y_valid)],
               verbose=100, early_stopping_rounds=100)

lgb_pred = lgbm_model.predict(x_valid_full)

# lgbm_roc = roc_auc_score(y_valid, lgb_pred)
print("accuracy  lgbm: ", accuracy_score(y_valid, lgb_pred))

#########################################################################################################

cat_m = CatBoostClassifier(n_estimators=5000,
                           random_state=1994,
                           eval_metric='MultiClass',
                           learning_rate=0.1, max_depth=5)

cat_m.fit(x_train_full, y_train,
          eval_set=[(x_valid_full, y_valid)],
          early_stopping_rounds=200,
          verbose=200)

cat_pred = cat_m.predict(x_valid_full)

print("accuracy  cat b: ", accuracy_score(y_valid, cat_pred))

##########################################################################################################

test_pred = cat_m.predict(test_full)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Trip_ID': test.index, 'Surge_Pricing_Type': test_pred[:, 0]})
output.to_csv('D:/Data Analytics - Personal/Analytics Vidhya/Cab Service/catm.csv', index=False)
