import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, precision_recall_fscore_support,classification_report
import seaborn as sns
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('dat''a_train.csv')
df.drop(["id"], axis=1, inplace=True)

print(df.isnull().sum().max())
null_df = df.isnull().sum()
null_index = []
for index, value in null_df.iteritems():
    if not value:
        null_index.append(index)
null_df.drop(null_index, inplace=True)

# df.dropna(inplace=True)

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
# plt.hist(num18)

num22 = df["num22"].values
num22 = [value for value in num22 if not np.isnan(value)]
# num18 = num18.reshape((-1,1))
# num18 = num18[~np.isnan(num18).any(axis=1)]
# plt.hist(num22)

miss_to_mean = ["num18", "num22"]

df.drop(drop_cols, axis = 1, inplace=True)

for col in miss_to_mean:
    df[col].fillna((df[col].mean()), inplace=True)

df.dropna(inplace=True)
cat_cols = [col for col in df.columns.values if col.startswith("cat")]
for col in cat_cols:
    df[cat_cols] = df[cat_cols].astype("category")

# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Entire DataFrame
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20})
# plt.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)


pca = PCA(n_components=2)
df_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(df_pca, y_train)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_res, y_res)
predicted = clf.predict(X_test_pca)

conf_1 = confusion_matrix(y_test, predicted)
recall_1 = recall_score(y_test, predicted)
precision_1 = precision_score(y_test, predicted)
f1_score = precision_recall_fscore_support(y_test, predicted)
class_report = classification_report(y_test, predicted)
# importance = clf.feature_importances_
# np.round(importance, 2)
print(conf_1)
print(f1_score)
print(recall_1)
print(precision_1)
print(class_report)
print(clf.get_params())

cv = GridSearchCV(clf, param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]})
cv.fit(X_res, y_res)
cv.best_params_

predicted = cv.predict(X_test_pca)

conf_1 = confusion_matrix(y_test, predicted)
recall_1 = recall_score(y_test, predicted)
precision_1 = precision_score(y_test, predicted)
f1_score = precision_recall_fscore_support(y_test, predicted)
class_report = classification_report(y_test, predicted)
# importance = clf.feature_importances_
# np.round(importance, 2)
print(conf_1)
print(f1_score)
print(recall_1)
print(precision_1)
print(class_report)
print(clf.get_params())
