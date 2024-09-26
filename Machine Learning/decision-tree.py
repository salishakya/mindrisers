import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import numpy as np
from sklearn.tree import plot_tree

data = pd.read_csv("../Week6-ML/csv/data.csv")
data.head()
data.isnull().sum()

data.columns

data.drop(columns="Unnamed: 0", inplace=True)

data["Age"].fillna(value=data["Age"].median(), inplace=True)

data["Embarked"].mode()[0]

data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

# [0]: Since .mode() can return multiple values as a Series, [0] selects the first mode (the most frequent value).

data["Fare"].fillna(data["Fare"].median(), inplace=True)


data = pd.get_dummies(data, columns=["Embarked"], drop_first=True)

# Here's a breakdown of what it does:

# pd.get_dummies(): This function converts categorical values into dummy/indicator variables (one-hot encoding).
# data: The input DataFrame that contains a column 'Embarked' with categorical values.
# columns=['Embarked']: Specifies that only the 'Embarked' column should be transformed into dummy variables.
# drop_first=True: Drops the first category to avoid multicollinearity (when using these dummy variables in models like linear regression), ensuring that the remaining dummy variables can represent all categories without redundancy.

# if the 'Embarked' column had three categories: S, C, and Q, using this code would create two new columns (e.g., Embarked_C and Embarked_Q), and the dropped category (S) would be represented when both new columns are 0.

data["Sex"].unique()

# map categorical into 1 and 0

data["Sex"] = data["Sex"].map({"male": 1, "female": 0})

# Standardization is important in machine learning to ensure that features are on a similar scale

data[["Age", "Fare"]]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[["Age", "Fare"]] = scaler.fit_transform(data[["Age", "Fare"]])

data[["Age", "Fare"]]

# Drop columns that won't be used in the model
data.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

data.isnull().sum()

# Fill missing Survived values with the most frequent value (mode)

data["Survived"].fillna(data["Survived"].mode()[0], inplace=True)

# Split data into X and y
y = data["Survived"]
X = data.drop(columns=["Survived"])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply Lasso to perform feature selection
lasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train)

# LassoCV: This is a Lasso regression model with built-in cross-validation to find the best value for the regularization parameter (alpha). The Lasso algorithm works by shrinking some coefficients to zero, which can be used for feature selection.
# cv=5: This specifies 5-fold cross-validation to tune the regularization parameter.
# .fit(X_train, y_train): Fits the Lasso model to the training data X_train (the features) and y_train (the target variable).

# Get selected features
coef = np.where(lasso.coef_ != 0)[0]

# lasso.coef_: After training, this attribute contains the coefficients of the Lasso model for each feature. If a feature's coefficient is 0, it means Lasso deemed it irrelevant.
# np.where(lasso.coef_ != 0)[0]: This selects the indices of features where the coefficients are not zero, meaning those features were selected by Lasso.
selected_features = X.columns[coef]

# X.columns[coef]: Using the indices of non-zero coefficients (coef), this maps back to the actual feature names that were selected by Lasso.
print("Selected features by Lasso:", selected_features)

# Lasso performs both regularization and feature selection. By setting some coefficients to zero, it automatically removes irrelevant or redundant features, making it useful for identifying important features in your dataset.

# Split data into X and y
X = data[["PassengerId", "Pclass", "Sex"]]
y = data["Survived"]

classifier = DecisionTreeClassifier(random_state=42).fit(X=X, y=y)
plot_tree(
    decision_tree=classifier,
    max_depth=3,
    feature_names=X.columns,
    class_names=["Not Survived", "Survived"],
    filled=True,
)
