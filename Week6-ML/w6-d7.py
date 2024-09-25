import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./csv/data.csv")
data.head()
data.isnull().sum()

data.columns

sns.countplot(data=data, x="Survived")
plt.title("Survival count")
plt.show()

sns.histplot(data["Age"], kde=True)
plt.title("Age Distrubution")
plt.show()

sns.boxplot(x="Survived", y="Age", data=data)
plt.title("Survival by Age")
plt.show()

sns.boxplot(x="Survived", y="Fare", data=data)
plt.title("Survival by Fare")
plt.show()

sns.countplot(x="Pclass", hue="Survived", data=data)
plt.title("Survival by Passenger Class")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=data)
plt.title("Survival by Gender")
plt.show()

data.drop(columns="Unnamed: 0", inplace=True)

data["Age"].fillna(value=data["Age"].median(), inplace=True)

data["Embarked"].mode()[0]

data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)

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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import numpy as np

# Split data into X and y
y = data["Survived"]
X = data.drop(columns=["Survived"])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply Lasso to perform feature selection
lasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train)

# Get selected features
coef = np.where(lasso.coef_ != 0)[0]
selected_features = X.columns[coef]
print("Selected features by Lasso:", selected_features)

# Split data into X and y
X = data[["PassengerId", "Pclass", "Sex"]]
y = data["Survived"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.linear_model import LogisticRegression

# Create the logistic regression model
model = LogisticRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
# Predicting the results for test set
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Plot confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
# Add labels and titles
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
# Show the plot
plt.show()

# see how much each feature (like Age or Fare) impacts the model’s prediction.

# model.coef_: After training a model (like linear regression), it calculates a number (called a "coefficient") for each feature. This number tells you how much that feature affects the model's predictions.

coefficients = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_[0]})
print(coefficients)

# In the output, each feature will have a number next to it. Positive numbers mean that as the feature increases, the prediction increases. Negative numbers mean that as the feature increases, the prediction decreases.

# Here, sex ,PassengerId, Pclass increase means survival decreases.

import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
print(result.summary())

# check predictions

# Create a DataFrame for predictions
predictions_df = pd.DataFrame(
    {
        "PassengerId": data.loc[
            X_test.index, "PassengerId"
        ],  # Use the indices to match
        "Predicted_Survived": y_pred,
    }
)
print(predictions_df)

# Merge predictions with the original DataFrame
final_df = data.merge(predictions_df, on="PassengerId", how="inner")

print(final_df[["PassengerId", "Survived", "Predicted_Survived"]].head())

# Save the final DataFrame to a CSV file
final_df.to_csv("./csv/titanic_with_predictions.csv", index=False)
