# decision tree in scikit

import pandas as pd

loan_data = pd.read_csv("./csv/loan_data.csv")
loan_data.info()
loan_data.head()

loan_data["emp_title"].fillna(loan_data["emp_title"].mode()[0], inplace=True)
loan_data["emp_length"].fillna(loan_data["emp_length"].mode()[0], inplace=True)
loan_data["title"].fillna(loan_data["title"].mode()[0], inplace=True)
loan_data["revol_util"].fillna(loan_data["revol_util"].median(), inplace=True)
loan_data["mort_acc"].fillna(loan_data["mort_acc"].median(), inplace=True)
loan_data["pub_rec_bankruptcies"].fillna(
    loan_data["pub_rec_bankruptcies"].median(), inplace=True
)

cat_columns = loan_data.select_dtypes(include=["object", "category"]).columns.tolist()
num_columns = loan_data.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Initialize the label encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# Applying label encoding to each of the categorical columns
for col in cat_columns:
    loan_data[col] = le.fit_transform(loan_data[col])

# Now, the dataset should be preprocessed and ready for feature
# selection and modeling
loan_data.info()
loan_data.head()

# Missing values were filled with either the mode (for categorical
# columns) or the median (for numerical columns).
# Categorical columns were label-encoded to numerical values for
# compatibility with machine learning algorithms

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

X = loan_data.drop(
    columns=["loan_status", "emp_title", "issue_d", "earliest_cr_line", "address"]
)  # Dropping non-relevant columns
y = loan_data["loan_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

feature_importances = tree_clf.feature_importances_

feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": feature_importances}
)

# Sort the DataFrame by importance in descending order
feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
).reset_index(drop=True)
feature_importance

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# True Positives (TP): 51,184
# True Negatives (TN): 4,592
# False Positives (FP): 10,985
# False Negatives (FN): 12,445

# Tune this model or try other machine learning algorithms to improve the results

#  The main hyperparameters to tune for Decision Trees are:

# max_depth: Maximum depth of the tree (controls overfitting).
# min_samples_split: Minimum number of samples required to split an internal node.
# min_samples_leaf: Minimum number of samples required to be at a leaf node.
# criterion: The function to measure the quality of a split (e.g., "gini" or "entropy").

#  Grid Search Cross-Validation

# Importing the necessary library for hyperparameter tuning
from sklearn.model_selection import GridSearchCV

# Setting up the hyperparameters for tuning
param_grid = {
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"],
}

# Setting up the GridSearchCV to find the best parameters

grid_search = GridSearchCV(
    estimator=tree_clf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1,
)

# Fitting the grid search to the data
grid_search.fit(X_train, y_train)
# Get the best parameters from the grid search
best_params = grid_search.best_params_
# Train a new Decision Tree with the best parameters
best_tree_clf = grid_search.best_estimator_
# Evaluating the optimized model
y_pred_tuned = best_tree_clf.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_conf_matrix = confusion_matrix(y_test, y_pred_tuned)

best_params
tuned_accuracy
tuned_conf_matrix

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(tuned_conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# GridSearchCV is a powerful tool in scikit-learn that automates the process of finding the best hyperparameters for a given model. It searches over a set of specified parameter values, fitting the model on different combinations, and selecting the best set of parameters based on cross-validation performance.

# param_grid: A dictionary specifying the hyperparameters to tune and the values to try for each.

# "max_depth": [10, 20, 30, None]: This defines the maximum depth of the decision tree. It will try trees with depths of 10, 20, 30, and no limit (None).

# "min_samples_split": [2, 10, 20]: This sets the minimum number of samples required to split an internal node. It will try 2, 10, and 20.

# "min_samples_leaf": [1, 5, 10]: This controls the minimum number of samples required to be at a leaf node.

# "criterion": ["gini", "entropy"]: These are the functions used to measure the quality of a split. The model will be tested with both Gini impurity and entropy.

# Gini impurity measuring how mixed up the data is in a particular group (or node) in a decision tree.
# If all the items in the group belong to the same class (for example, all apples or all oranges), Gini impurity is 0—the group is perfectly pure.
# If the group has a mix of different classes (like half apples, half oranges), the Gini impurity increases. A value of 0.5 means the group is split equally, making it very impure.

# Entropy is like measuring the uncertainty or disorder in the group.
# If a group is made up of just one kind of thing (all apples or all oranges), entropy is 0—there’s no uncertainty about what’s in the group.
# If the group is half apples and half oranges, entropy is at its highest because the group is most uncertain or mixed up.

# Both Gini impurity and entropy help the decision tree figure out where to split the data. The goal is to make groups that are as pure as possible.

# The tree looks at different ways to split the data and chooses the split that makes the groups less mixed (lower Gini impurity or entropy).

# In simple terms, both Gini and entropy measure how messed up or uncertain a group of data is, and decision trees aim to reduce this messiness at every step.

# cv: The number of cross-validation folds. Here, cv=3 means that the data will be split into 3 folds, and the model will be trained and validated on these folds to evaluate its performance.

# scoring: This defines the metric to evaluate the performance of the model. In this case, scoring="accuracy" means the grid search will optimize for accuracy.

# n_jobs: Specifies the number of jobs (threads) to run in parallel. n_jobs=-1 uses all available CPU cores, speeding up the process.

# verbose: Controls the verbosity of the output. verbose=1 will print progress during the search, helping you monitor the process.
