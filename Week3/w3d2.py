import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# merge and join - pd.merge() conditions sabai bracket bhitra huncha

df1 = pd.DataFrame(
    {
        "employee": ["Bob", "Jake", "Lisa", "Sue"],
        "group": ["Accounting", "Engineering", "Engineering", "HR"],
    }
)

df2 = pd.DataFrame(
    {
        "employee": ["Lisa", "Bob", "Jake", "Sue"],
        "hire_date": [2004, 2008, 2011, 2009],
    }
)

print(df1)
print("-" * 20)
print(df2)

df3 = pd.merge(df1, df2)
print(df3)

df4 = pd.DataFrame(
    {
        "group": ["Accounting", "Engineering", "HR"],
        "supervisor": ["Carly", "Guido", "Steve"],
    }
)
print(df3)
print("-" * 20)
print(df4)

pd.merge(df3, df4)

pd.merge(df4, df3)

df5 = pd.DataFrame(
    {
        "group": ["Accounting", "Accounting", "Engineering", "Engineering", "HR", "HR"],
        "skills": [
            "math",
            "spreadsheets",
            "coding",
            "linux",
            "spreadsheets",
            "organization",
        ],
    }
)

print(df5)

pd.merge(df1, df5)

# specifications in merge

print(df1)
print("-" * 20)
print(df2)

pd.merge(df1, df2, on="employee")

df6 = pd.DataFrame(
    {"name": ["Bob", "Jake", "Lisa", "Sue"], "salary": [70000, 80000, 120000, 90000]}
)

print(df1)
print("-" * 40)
print(df6)

pd.merge(df1, df6, left_on="employee", right_on="name")

# name same cha so drop garda huncha
pd.merge(df1, df6, left_on="employee", right_on="name").drop("name", axis=1)

df1a = df1.set_index("employee")
df2a = df2.set_index("employee")

df7 = pd.DataFrame(
    {"name": ["peter", "paul", "mary"], "food": ["fish", "beans", "bread"]},
    columns=["name", "food"],
)

df8 = pd.DataFrame(
    {"name": ["mary", "joseph"], "drink": ["wine", "beer"]},
    columns=["name", "drink"],
)

pd.merge(df7, df8)

pd.merge(df7, df8, how="inner")

pd.merge(df7, df8, how="outer")

pd.merge(df7, df8, how="left")

pd.merge(df7, df8, how="right")

# outer is union, inner is intersection

# left le left ko table ko primary key hercha and same for right

df9 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"], "rank": [1, 2, 3, 4]})

df10 = pd.DataFrame({"name": ["Bob", "Jake", "Lisa", "Sue"], "rank": [2, 1, 3, 4]})

pd.merge(df9, df10, on="name")

pd.merge(df9, df10, on="name", suffixes=["_left", "_right"])

# # download state population data
# !curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-population.csv

# # download state areas data
# !curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-areas.csv

# !curl -O https://raw.githubusercontent.com/jakevdp/data-USstates/master/state-abbrevs.csv

pop = pd.read_csv("state-population.csv")
areas = pd.read_csv("state-areas.csv")
abbrevs = pd.read_csv("state-abbrevs.csv")

print(pop.head())
print("-" * 40)
print(areas.head())
print("-" * 40)
print(abbrevs.head())

merged = pd.merge(
    pop, abbrevs, how="outer", left_on="state/region", right_on="abbreviation"
)
merged = merged.drop(columns="abbreviation")
print(merged)

merged1 = pd.merge(
    pop, abbrevs, how="inner", left_on="state/region", right_on="abbreviation"
)
print(merged1)

merged.head()

merged.isnull().any()  # any le chai tyo column ma cha ki chaina vanera dekhaucha
