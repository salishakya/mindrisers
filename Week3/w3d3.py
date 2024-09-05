import pandas as pd

pop = pd.read_csv("state-population.csv")
areas = pd.read_csv("state-areas.csv")
abbrevs = pd.read_csv("state-abbrevs.csv")

merged = pd.merge(
    pop, abbrevs, how="outer", left_on="state/region", right_on="abbreviation"
)
merged = merged.drop(columns="abbreviation")
print(merged)

merged.isnull().any()

# hijo ko sabai run gareko ho yo

# aba chai we are checking the null values in population since 'True' ako cha

merged[merged["population"].isnull()].head()

merged.isnull().sum()

merged.loc[merged["state"].isnull(), "state/region"].unique()

merged[merged["state"].isnull(), "state/region"].unique()

merged.loc[merged["state/region"] == "PR", "state"] = "Puerto Rico"
merged.loc[merged["state/region"] == "USA", "state"] = "United States"

merged.isnull().any()

final = pd.merge(merged, areas, on="state", how="left")
final.head()

final.isnull().any()

final[final["area (sq. mi)"].isnull()].head()

final["state"][final["area (sq. mi)"].isnull()].unique()

# which of the states have area null?

final.dropna(inplace=True)
final.isnull().any()

# PR ko mistake vako cha, PR ko NaN values lai chai replaced with it's mean

data2010 = final.query('year == 2010 & ages == "total"')
data2010.head()

data2010.set_index("state", inplace=True)

# inplace = True le chai orginal data mei change garcha

density = data2010["population"] / data2010["area (sq. mi)"]

density.sort_values(ascending=False, inplace=True)
density.head()
# ranking density
density.tail()

data2010

data2010.loc["density"] = density

# add density as column at home

import seaborn as sb

planets = sb.load_dataset("planets")
planets.shape

# planets is already available in the library

planets.head()

import numpy as np

rng = np.random.RandomState(42)

ser = pd.Series(rng.rand(5))
ser

planets.describe()

planets.dropna().describe()
# drop the nan and describe the dataset

planets

planets.groupby("method")

planets_series_groupby = planets.groupby("method")["orbital_period"]
planets_series_groupby

planets_series_groupby.median()

# groupby is a lazy evaluation, method bina kaam nei gardaina

for method, group in planets.groupby("method"):
    print("{0:30s} shape={1}".format(method, group.shape))

from nbconvert import NotebookExporter
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat

# Load the notebook
with open("your_notebook.ipynb") as file:
    notebook = nbformat.read(file, as_version=4)

# Execute the notebook
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(notebook, {"metadata": {"path": "./"}})

# Save the executed notebook
with open("executed_notebook.ipynb", "w") as file:
    nbformat.write(notebook, file)
