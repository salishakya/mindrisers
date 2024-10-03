# Random Forest Algorithm

# Objective: Build a Random Forest model to predict the loan status (Loan_Status) in the test dataset.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

data = pd.read_csv("./csv/loan_data.csv")
data.head()

data.info()

data.describe()

cat_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
num_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

data[num_columns].hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

cat_columns.__len__()

num_columns.__len__()

# Categorical Features Distribution
# for feature in cat_columns:
#     sns.countplot(x=feature, data=data)
#     plt.title(f"{feature} Distribution")
#     plt.show()

for feature in num_columns:
    sns.boxplot(x="loan_status", y=feature, data=data)
    plt.title(f"Loan_Status vs {feature}")
    plt.show()

#  Loan_Status vs Categorical Features

# for feature in cat_columns:
#     sns.countplot(x=feature, hue="loan_status", data=data)
#     plt.title(f"Loan_Status vs {feature}")
#     plt.show()

data.isnull().sum()

# Fill missing values in numerical columns with the median
for col in num_columns:
    data[col].fillna(data[col].median(), inplace=True)
# Fill missing values in categorical columns with the mode
for col in cat_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
data["grade"] = data["grade"].map(grade_mapping)

data["sub_grade"].unique()

data["sub_grade"] = data["sub_grade"].apply(lambda x: int(x[1]))

# .apply(lambda x: int(x[1])): The apply() function applies a given operation (in this case, a lambda function) to each element of the sub_grade column.

# lambda x: int(x[1]): This lambda function takes each value x in the sub_grade column, extracts the second character (x[1]), and converts it into an integer using the int() function.
# Example: If x is the string 'B2', x[1] will be '2', and int(x[1]) will convert it to the integer 2.

data["loan_status"].replace({"N": 0, "Y": 1}, inplace=True)

data.head()

"loan_status" in data.columns

new_cat_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()

new_cat_columns

for col in new_cat_columns:
    if data[col].nunique() == 2:
        print(data[col].name)

# One-Hot Encoding for Nominal Variables
nominal_columns = [
    "term",
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
]

data = pd.get_dummies(data, columns=nominal_columns, drop_first=True)

# Convert 'issue_d' to datetime and extract useful features
data["issue_d"] = pd.to_datetime(data["issue_d"], format="%b-%Y")
data["issue_year"] = data["issue_d"].dt.year
data["issue_month"] = data["issue_d"].dt.month
data.drop("issue_d", axis=1, inplace=True)

# Convert 'earliest_cr_line' to datetime and extract useful features
data["earliest_cr_line"] = pd.to_datetime(data["earliest_cr_line"], format="%b-%Y")
data["earliest_cr_line_year"] = data["earliest_cr_line"].dt.year
data.drop("earliest_cr_line", axis=1, inplace=True)

data.head()

data.info()

new_new_cat_columns = data.select_dtypes(
    include=["object", "category"]
).columns.tolist()

cat_data = data[new_new_cat_columns]

cat_data.head()

# One-Hot Encoding for Nominal Variables
# new_nominal_columns = ["emp_title", "loan_status", "title", "address"]

# data = pd.get_dummies(data, columns=["loan_status"], drop_first=True)

#  Feature Engineering

# Calculate the mean of annual_inc, ignoring zero values
mean_annual_inc = data.loc[data["annual_inc"] != 0, "annual_inc"].mean()
# Replace zero annual_inc with the mean annual_inc
data["annual_inc"] = data["annual_inc"].replace(0, mean_annual_inc)
# Calculate income_loan_ratio
data["income_loan_ratio"] = data["loan_amnt"] / data["annual_inc"]

data.info()

new_new_cat_columns.remove("loan_status")

le = LabelEncoder()
# Applying label encoding to each of the categorical columns
for col in new_new_cat_columns:
    data[col] = le.fit_transform(data[col])

data.info()

# everything except the loan status is numerical

# Correlation Clustering:

import scipy.cluster.hierarchy as sch

# Perform hierarchical clustering on the correlation matrix
corr_matrix = data.corr()
corr_linkage = sch.linkage(sch.distance.pdist(corr_matrix), method="complete")
# Plot the dendrogram to visualize clusters
plt.figure(figsize=(10, 7))
dendrogram = sch.dendrogram(corr_linkage, labels=corr_matrix.columns, leaf_rotation=90)
plt.title("Feature Clustering Dendrogram")
plt.show()
# You can then manually select features from each cluster

# We don't want this "Correlation Cluster". It's weird that we need numerical but pdf ma j pani chalcha

data.drop(columns=["emp_title", "emp_length", "title", "address"], inplace=True)

# Check for infinite values
print(data.isin([np.inf, -np.inf]).sum())

y = data["loan_status"]
X = data.drop(["loan_status"], axis=1)

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=X.columns)

# Sort the importances in descending order and select the top 10 features
top_10_importances = importances.sort_values(ascending=False).head(10)

# Plot the top 10 feature importances
plt.figure(figsize=(10, 6))
top_10_importances.plot(kind="barh", color="skyblue")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# Display the top 10 feature importances
print(top_10_importances)

data["loan_status"].value_counts()

# SMOTE is a technique that generates synthetic samples for the minority class to balance the dataset.
sm = SMOTE(random_state=42)

# It resamples the dataset, generating new synthetic samples for the minority class here it's "Charged off"
X_res, y_res = sm.fit_resample(X, y)
# Check the new class distribution
print(pd.Series(y_res).value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict the target labels for the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

param_grid = {
    "n_estimators": [100, 200],  # Fewer options
    "max_depth": [6, None],  # Fewer depths to explore
    "max_features": ["sqrt"],  # Use just 'sqrt'
    "min_samples_split": [2, 5],  # Fewer split points
    "bootstrap": [True],  # Only one option for bootstrap
}

random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_grid,
    n_iter=8,
    cv=3,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
)
random_search.fit(X_train, y_train)
print("Best Parameters:", random_search.best_params_)

# RandomizedSearchCV is a method that performs randomized hyperparameter tuning. It searches over a subset of possible parameter values and selects the best combination based on performance. It’s more efficient than GridSearchCV, which exhaustively checks all parameter combinations.
# This specifies the number of iterations to run, meaning RandomizedSearchCV will try 8 random combinations of parameters from param_grid
# Cross-validation strategy: 3-fold cross-validation is used. The data will be split into 3 parts, and the model will be trained and evaluated 3 times, each time using a different part for validation and the remaining parts for training.
# scoring="roc_auc": This specifies the scoring metric used to evaluate the model's performance. "roc_auc" stands for the Area Under the Receiver Operating Characteristic Curve (ROC AUC). It’s a commonly used metric for classification problems, especially when dealing with imbalanced data.

# why not scoring = "accuracy"
# In imbalanced classification problems, ROC AUC (or other metrics like F1-score, precision, recall) is often preferred over accuracy because it gives a better sense of how well the model handles both the majority and minority classes. Accuracy may lead to overly optimistic results in such cases and doesn't capture the full picture of model performance.
# Example: If 95% of your data belongs to Class A and only 5% to Class B, a model that predicts Class A all the time would have 95% accuracy, but it would be completely useless in detecting Class B.

# Training the Model with Best Parameters
best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)
RandomForestClassifier(n_estimators=200, random_state=42)

# Split the data into train, validation, and test sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Predict on the validation set
y_pred = best_rf.predict(X_valid)
y_proba = best_rf.predict_proba(X_valid)[:, 1]  # Probabilities for ROC AUC

# the [:, 1] part is used to select the probabilities of the positive class from the predicted probabilities returned by the predict_proba method.
# predict_proba(X_valid), the Random Forest model returns a 2D array (or DataFrame) where each row corresponds to a sample in the input data (X_valid), and each column corresponds to a class label.

# The first column (index 0) contains the predicted probabilities of the negative class (e.g., loan_status = 0).
# The second column (index 1) contains the predicted probabilities of the positive class (e.g., loan_status = 1).

print(classification_report(y_valid, y_pred))
print("y probability", y_proba)

cm = confusion_matrix(y_valid, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.show()

cv_scores = cross_val_score(best_rf, X_res, y_res, cv=5, scoring="roc_auc")
print("Cross-Validation AUC Scores:", cv_scores)
print("Mean CV AUC Score:", cv_scores.mean())

# cross_val_score Function: This function is used to evaluate a model's performance using cross-validation.
# print("Cross-Validation AUC Scores:", cv_scores): This line prints the array of ROC AUC scores obtained from each of the 5 cross-validation folds. Each score represents the model's performance on one of the validation folds.
# print("Mean CV AUC Score:", cv_scores.mean()): This line calculates and prints the mean ROC AUC score across all 5 folds, providing a single summary metric that indicates the overall performance of the model.
