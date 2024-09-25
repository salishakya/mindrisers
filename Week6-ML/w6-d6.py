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

# OneHotEncoder(drop="first"):

# The OneHotEncoder is used to transform categorical data into a numeric format that a machine learning model can understand. It converts categories into binary vectors (one-hot encoding), where each category is represented as a unique combination of 0s and 1s.
# drop="first": When encoding a categorical feature, this parameter drops the first category. This avoids the dummy variable trap, which is a form of multicollinearity that can occur when one category can be predicted as a linear combination of others. By dropping one category, you make the features independent of each other.
# In this case, the satisfaction column is being encoded into a numeric form, likely from a binary categorical format like "satisfied" and "not satisfied." The first category (e.g., "not satisfied") will be dropped, and the other category ("satisfied") will be encoded as 1.

# fit_transform(df_subset[["satisfaction"]]):

# fit(): Learns the structure of the categorical data (i.e., determines how many unique categories exist in the satisfaction column).
# transform(): Converts the column into a binary one-hot encoded format.
# fit_transform() combines both steps, applying them in one operation. It learns the encoding from the satisfaction column and transforms the data into a numeric representation.
# Since the column is passed inside double square brackets (df_subset[["satisfaction"]]), it is treated as a DataFrame (not a Series), which is what OneHotEncoder expects.

# .toarray():

# The fit_transform() method returns a sparse matrix by default, which is memory efficient. However, the logistic regression model needs a dense matrix or array. Calling .toarray() converts the sparse matrix into a dense NumPy array, which stores the data in a standard format (1s and 0s).

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
# The line predicts the probability that each sample in the test set belongs to either class (e.g., "satisfied" or "not satisfied").
# predict_proba(): This method outputs the probability of each class for each observation in the test set X_test. Instead of just predicting a hard label (0 or 1), this method gives a probabilistic prediction.
# [[0.30, 0.70],  # 70% chance of being in class 1 (satisfied)
#  [0.80, 0.20],  # 20% chance of being in class 1 (satisfied)
#  [0.40, 0.60],  # 60% chance of being in class 1 (satisfied)
#  [0.90, 0.10],  # 10% chance of being in class 1 (satisfied)
#  [0.25, 0.75]]  # 75% chance of being in class 1 (satisfied)
#  After obtaining these probabilities, you can either:
#
# Set a threshold (usually 0.5) to classify the samples (e.g., if the probability for class 1 is above 0.5, classify it as "satisfied").
# Use the raw probabilities for more nuanced decision-making, such as ranking samples by their likelihood of being satisfied.

# By default, scikit-learn's LogisticRegression uses a 0.5 threshold to make predictions. This means:
# If the predicted probability for class 1 (e.g., "satisfied") is greater than or equal to 0.5, the model classifies the instance as class 1.
# If the predicted probability for class 1 is less than 0.5, it classifies the instance as class 0.

clf.predict(X_test)

print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))

cm = metrics.confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
