import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

df = pd.read_csv('data_train.csv')
df.drop(["Unnamed: 0", "Unnamed: 0.1", "id"], axis=1, inplace=True)

print(df.isnull().sum().max())
null_df = df.isnull().sum()
null_index = []
for index, value in null_df.iteritems():
    if not value:
        null_index.append(index)
null_df.drop(null_index, inplace=True)

df.dropna(inplace=True)

all_cols = list(df.columns.values)
del all_cols[-1]
cat_cols = [col for col in all_cols if col.startswith("cat")]
num_cols = [col for col in all_cols if not col.startswith("cat")]

null_df_percent = (null_df/len(df))*100
missing_cols = null_df.index.values

drop_cols = ["cat6", "cat8"]
remov_miss = ["num19", "num20", "cat1", "cat2", "cat3", "cat4", "cat5", "cat10", "cat12"]

num18 = df["num18"].values
num18 = [value for value in num18 if not np.isnan(value)]
# num18 = num18.reshape((-1,1))
# num18 = num18[~np.isnan(num18).any(axis=1)]
plt.hist(num18)

num22 = df["num22"].values
num22 = [value for value in num22 if not np.isnan(value)]
# num18 = num18.reshape((-1,1))
# num18 = num18[~np.isnan(num18).any(axis=1)]
plt.hist(num22)

miss_to_mean = ["num18", "num22"]

df.drop(drop_cols, axis = 1, inplace=True)

for col in miss_to_mean:
    df[col].fillna((df[col].mean()), inplace=True)

df.dropna(inplace=True)
cat_cols = [col for col in df.columns.values if col.startswith("cat")]
for col in cat_cols:
    df[cat_cols] = df[cat_cols].astype("category")

y = df["target"]
y.reset_index(drop = True, inplace=True)
df.drop("target", axis=1, inplace=True)

df_cat = df[cat_cols]
df_cat.reset_index(drop = True, inplace=True)
df_num = df.drop(cat_cols, axis=1)
df_num_cols = df_num.columns.values

df_num = pd.DataFrame(StandardScaler().fit_transform(df_num), columns=df_num_cols)
df_num.reset_index(drop = True, inplace=True)

X = pd.concat([df_num, df_cat], axis=1)

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

