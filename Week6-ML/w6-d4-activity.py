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

# For multiple linear regression to work well, some assumptions should be satisfied/ some prerequisites should be met.

# 1. Linearity

# The assumption of linearity means that the relationship between the dependent variable (y) and the independent variables (X) should be linear. If this assumption holds, changes in the predictors (independent variables) should result in proportional changes in the dependent variable.

# Why it’s important: If the relationship isn’t linear, the model may give inaccurate predictions.
# How to check: Plot the independent variables against the dependent variable using a scatter plot. If the points roughly form a straight line, the linearity assumption is satisfied.

data_X["Radio"].shape
data_y["Sales"].shape

# X can be one of your independent variables, and y is your dependent variable
sns.scatterplot(x=data_X["Radio"], y=data_y["Sales"])
plt.title("Scatter plot of Radio vs Sales")
plt.show()

sns.scatterplot(x=data_X["Social_Media"], y=data_y["Sales"])
plt.title("Scatter plot of Social Media vs Sales")
plt.show()

# Here the relationship is linear between Radio and Sales but not for social media

# Model assumption: Independence
# The independent observation assumption states that each observation in the dataset is independent. As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# Model assumption: Normality

import scipy.stats as stats

# Q-Q plot
stats.probplot(model.resid, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# The residuals (errors) should follow a normal distribution. This is important for reliable hypothesis testing and confidence intervals.

# Why it’s important: Non-normally distributed residuals can lead to biased estimates.
# How to check: You can use a Q-Q plot (quantile-quantile plot) or a histogram of residuals. In a Q-Q plot, if the points fall along a straight line, the residuals are normally distributed.

plt.hist(model.resid)

# Model assumption: No multicollinearity

# Multicollinearity happens when two or more independent variables are highly correlated with each other. This makes it difficult to determine the effect of each independent variable on the dependent variable.

# Why it’s important: Multicollinearity can lead to unstable estimates of the regression coefficients, making it hard to interpret the model.
# How to check: Use the Variance Inflation Factor (VIF). A VIF value greater than 10 indicates high multicollinearity.

X = data[["Radio", "Social_Media"]]
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
df_vif = pd.DataFrame(vif, index=X.columns, columns=["VIF"])
df_vif
