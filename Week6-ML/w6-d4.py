import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins")
penguins.columns
penguins.head()

penguins = penguins[["body_mass_g", "bill_length_mm", "sex", "species"]]
penguins.columns = ["body_mass_g", "bill_length_mm", "gender", "species"]
penguins.dropna(inplace=True)
penguins.reset_index(inplace=True, drop=True)

penguins_X = penguins[["bill_length_mm", "gender", "species"]]
penguins_y = penguins[["body_mass_g"]]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    penguins_X, penguins_y, test_size=0.3, random_state=42
)

# C is categorial variables = text strings
ols_formula = "body_mass_g ~ bill_length_mm + C(gender) + C(species)"

from statsmodels.formula.api import ols

ols_data = pd.concat([X_train, y_train], axis=1)

OLS = ols(formula=ols_formula, data=ols_data)
model = OLS.fit()

model.summary()

sns.regplot(x="bill_length_mm", y="body_mass_g", data=ols_data)

sns.regplot(x="gender", y="body_mass_g", data=ols_data)

# 1. Plotting the regression line for bill_length_mm vs body_mass_g
plt.figure(figsize=(10, 6))
sns.regplot(x="bill_length_mm", y="body_mass_g", data=ols_data)
plt.title("Regression of Body Mass on Bill Length")
plt.show()

# 2. Visualizing the effect of gender on body_mass_g
plt.figure(figsize=(10, 6))
sns.boxplot(x="gender", y="body_mass_g", data=ols_data)
plt.title("Body Mass by Gender")
plt.show()

# 3. Visualizing the effect of species on body_mass_g
plt.figure(figsize=(10, 6))
sns.boxplot(x="species", y="body_mass_g", data=ols_data)
plt.title("Body Mass by Species")
plt.show()
