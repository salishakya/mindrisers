# Please download the Credit Card Fraud detection csv file from vibver and proceed it to find whether a transaction is fraud or legit using:

# Logistic Regression
# Decision Tree Classifier
# Random Forest Classifier

# Please download the Credit-Card Fraud detection csv file from viber and proceed it to find whether a transaction is fraud or legit using logistic regression, decision tree classifier and random forest classifier.
# Compare the result and show which performed better in this classification task.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
