# Please download the Credit Card Fraud detection csv file from vibver and proceed it to find whether a transaction is fraud or legit using:

# Logistic Regression
# Decision Tree Classifier
# Random Forest Classifier

# Please download the Credit-Card Fraud detection csv file from viber and proceed it to find whether a transaction is fraud or legit using logistic regression, decision tree classifier and random forest classifier.
# Compare the result and show which performed better in this classification task.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

data = pd.read_csv("../csv/credit_card_fraud.csv")
data.head()

data.info()

data.isnull().sum()

# Check for infinite values
print(data.isin([np.inf, -np.inf]).sum())

data.shape

mean_transaction_amount = data.loc[
    data["Transaction_Amount"] != 0, "Transaction_Amount"
].mean()

data["Transaction_Amount"].fillna(value=mean_transaction_amount, inplace=True)

plt.hist(x=data["Customer_Age"])
plt.show()

data["Customer_Age"].fillna(value=data["Customer_Age"].median(), inplace=True)

data["Customer_Income_Bracket"].fillna(
    value=data["Customer_Income_Bracket"].mode()[0], inplace=True
)

cat_columns = data.select_dtypes(include=["object"]).columns.tolist()
num_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

data[cat_columns]

one_hot_columns = [
    "Merchant_Category",
    "Merchant_Location",
    "Payment_Type",
    "Device_Type",
    "Customer_Country",
    "Customer_Gender",
]

data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

new_cat_columns = data.select_dtypes(include=["object"]).columns.tolist()

le = LabelEncoder()
# Applying label encoding to each of the categorical columns
for col in new_cat_columns:
    data[col] = le.fit_transform(data[col])

# Split data into X and y
X = data.drop("Fraud_Flag", axis=1)
y = data["Fraud_Flag"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

param_grid = {
    "C": [0.1, 1, 10],
    "penalty": ["l2"],
    "solver": ["liblinear"],
    "max_iter": [100, 200],
}

# Instantiate Logistic Regression
log_reg = LogisticRegression()

# Hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(
    estimator=log_reg, param_grid=param_grid, cv=5, scoring="accuracy"
)
grid_search.fit(X_train_resampled, y_train_resampled)

# Best hyperparameters
print(f"Best hyperparameters: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy}")

print(
    f"Logistic Regression Classification Report: {classification_report(y_test, y_pred)}"
)

# Instantiate Decision Tree
tree_model = DecisionTreeClassifier()

# Grid for Decision Tree
param_grid_tree = {
    "max_depth": [10, 20],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5],
    "criterion": ["gini", "entropy"],
}

# Hyperparameter tuning using GridSearchCV
grid_search_tree = GridSearchCV(
    estimator=tree_model, param_grid=param_grid_tree, cv=5, scoring="accuracy"
)
grid_search_tree.fit(X_train_resampled, y_train_resampled)

# Best hyperparameters for Decision Tree
print(f"Best hyperparameters for Decision Tree: {grid_search_tree.best_params_}")

best_tree_model = grid_search_tree.best_estimator_
y_pred = best_tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy}")

print(f"Decision Tree Classification Report: {classification_report(y_test, y_pred)}")


# Instantiate Random Forest
random_forest_model = RandomForestClassifier()

# Grid for Random Forest
random_forest_param = {
    "max_depth": [10, 20, 30, None],
    "min_samples_split": [2, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"],
}

# Hyperparameter tuning using GridSearchCV
grid_search_random_forest = GridSearchCV(
    estimator=random_forest_model,
    param_grid=random_forest_param,
    cv=5,
    scoring="accuracy",
)
grid_search_random_forest.fit(X_train_resampled, y_train_resampled)

# Best hyperparameters for Random Forest
print(
    f"Best hyperparameters for Random Forest: {grid_search_random_forest.best_params_}"
)

best_random_forest_model = grid_search_random_forest.best_estimator_
y_pred = best_random_forest_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")

print(f"Random Forest Classification Report: {classification_report(y_test, y_pred)}")

# Copied from Manish :)

# separating Fraud and Not Fraud transaction

fraud_txn = data[data["Fraud_Flag"] == 1]
not_fraud_txn = data[data["Fraud_Flag"] == 0]

# Undersample majority class
majority_downsampled = not_fraud_txn.sample(50000)

# Combine minority class with downsampled majority class
train_data_balanced = pd.concat([majority_downsampled, fraud_txn])

# Separate features and target variable
X_train_balanced = train_data_balanced.drop("Fraud_Flag", axis=1)
y_train_balanced = train_data_balanced["Fraud_Flag"]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_train_balanced,
    y_train_balanced,
    test_size=0.3,
    random_state=42,
    stratify=y_train_balanced,
)

# I figured that the best model would probably be random forest so, I used it....
y_pred_d = best_random_forest_model.predict(X_test_d)

# Evaluate the model
print("\nClassification Report:\n", classification_report(y_test_d, y_pred_d))
