# Annotated follow-along guide_ Interpret multiple regression results with Python

# multicolinearity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data = pd.read_csv("./csv/marketing_sales.csv")
data.head()
data.columns

# • TV promotional budget (in “Low,” “Medium,” and “High” categories)
# • Social media promotional budget (in millions of dollars)
# • Radio promotional budget (in millions of dollars)
# • Sales (in millions of dollars)
# • Influencer size (in “Mega,” “Macro,” “Micro,” and “Nano” categories)

sns.pairplot(data=data)

tv_sales = data[["TV", "Sales"]]
tv_sales_mean = tv_sales["Sales"].mean()

influencer_sales = data[["Influencer", "Sales"]]
influencer_sales_mean = influencer_sales["Sales"].mean()

data.groupby("TV")["Sales"].mean()
data.groupby("Influencer")["Sales"].mean()

data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

data.columns

data.rename(columns={"Social Media": "Social_Media"}, inplace=True)

# Define the OLS formula.
# Create an OLS model.
# Fit the model.
# Save the results summary.
# Display the model results.

data_X = data[["TV", "Radio", "Social_Media", "Influencer"]]
data_y = data[["Sales"]]

X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.3, random_state=42
)

ols_formula = "Sales ~ Radio + C(TV)"

ols_data = pd.concat([X_train, y_train], axis=1)

ols_data.iloc[1]
data.iloc[1]

from statsmodels.formula.api import ols

OLS = ols(formula=ols_formula, data=ols_data)
model = OLS.fit()

model.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

X = data[["Radio", "Social_Media"]]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
df_vif = pd.DataFrame(vif, index=X.columns, columns=["VIF"])
df_vif
