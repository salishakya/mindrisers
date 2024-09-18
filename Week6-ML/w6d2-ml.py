# linear regression using ols

import pandas as pd
import seaborn as sns

penguins = sns.load_dataset("penguins")
# Examine first 5 rows of dataset
penguins.head()

penguins_sub = penguins[penguins["species"] != "Chinstrap"]
penguins_final = penguins_sub.dropna()
penguins_final.reset_index(inplace=True, drop=True)

sns.pairplot(penguins_final)

ols_data = penguins_final[["bill_length_mm", "body_mass_g"]]

# Write out formula
ols_formula = "body_mass_g ~ bill_length_mm"

from statsmodels.formula.api import ols

# Build OLS, fit model to data
OLS = ols(formula=ols_formula, data=ols_data)
model = OLS.fit()

model.summary()

sns.regplot(x="bill_length_mm", y="body_mass_g", data=ols_data)
