import pandas as pd
import numpy as np

rng = np.random.RandomState(42)
A = pd.DataFrame(rng.randint(0, 20, (2, 2)), columns=list("AB"))
print(A)

B = pd.DataFrame(rng.randint(0, 10, (2, 3)), columns=list("BAC"))
print(B)

A + B

A.add(B, fill_value=0)

# dataframe - series

A - A.iloc[0]

A = rng.randint(10, size=(3, 4))
A

A - A[0]

df = pd.DataFrame(A, columns=list("QRST"))
df

df - df.iloc[0]

# Dealing with missing values in python ( NaN(np ko) values and None (pythonko) object)

# nan safe mode

vals1 = np.array([1, None, 2, 3])
vals1
Vals2 = np.array([1, np.nan, 2, 3])
Vals2

Vals2.sum()
np.nansum(Vals2)
# nansum ignores nan values - it's the safe mode

np.nanmax(Vals2)

data = pd.Series([1, np.nan, "hello", None])

data

data.isnull()

data.isnull().sum()

data.notnull()

data[data.notnull()]

data.dropna()  # drop not available

data

data.dropna(inplace=True)

# reset index
data.reset_index()

df = pd.DataFrame([[1, 2, np.nan], [2, 3, 4], [5, 6, np.nan]])
df

df.dropna()
# dropna le row nei hatayo

df.dropna(axis=1)

df1 = pd.DataFrame([[1, 2, np.nan], [2, 3, 4], [np.nan, 6, 5]])
df1
df1.dropna(axis="columns")

df[3] = np.nan

df

df.dropna(axis="columns", how="all")

# sabai value NaN cha vaney chai drop gardincha

df.dropna(axis="rows", thresh=2)

df.dropna(axis="rows", thresh=3)  # keep roles with a minimum of 3 non-null values

df.fillna(0)
df.fillna(5)

df.fillna(method="ffill")

# method = "ffill" is now deprecated so using ffill function

df.ffill()

df.bfill()
