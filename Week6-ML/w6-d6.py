# Logistic Regression in Machine Learning

import numpy as np
import pandas as pd

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns

df_original = pd.read_csv("./csv/Invistico_Airline.csv")

df_original.head(10)

df_original.dtypes

df_original["satisfaction"].value_counts(dropna=False)

df_original["satisfaction"].value_counts()

df_original.isnull().sum()

df_subset = df_original.dropna(axis=0).reset_index(drop=True)
df_subset.head()

df_subset.columns

"Inflight entertainment" in df_subset.columns

df_subset = df_subset.astype({"Inflight entertainment": float})

# Convert the categorical column satisfaction into numeric

df_subset["satisfaction"] = (
    OneHotEncoder(drop="first").fit_transform(df_subset[["satisfaction"]]).toarray()
)

df_subset.head(10)

X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

clf = LogisticRegression().fit(X_train, y_train)

print(clf)

clf.intercept_

sns.regplot(
    x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None
)

y_pred = clf.predict(X_test)
y_pred

# Use predict_proba to output a probability.

clf.predict_proba(X_test)

clf.predict(X_test)

print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))

cm = metrics.confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
