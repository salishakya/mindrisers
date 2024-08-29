import pandas as pd
import numpy as np

data = pd.Series([0.25, 0.5, 0.75, 1.0])
data1 = np.array([0.25, 0.5, 0.75, 1.0])
print("Series: ", data)
print("Array: ", data1)

# 1D arrays are series
# 2D arrays are dataframes
# pandas is mostly for data cleaning and ML

# array banauni vaye numpy nei use garnu parcha

# Series le tabular structure ma rakhdyo

data.values

# series ko value sabai array mei store huncha

data.index

data[2]

# numpy array defines implicitly
# pandas series is explicitly defined

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d"])
data

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d"])
data

data["c"]

population_dic = {
    "California": 38,
    "Texas": 26,
    "New York": 20,
    "Florida": 19,
    "Illinois": 13,
}

population = pd.Series(population_dic)
population

# keys lai index ma halyo ani values side ma

# explicit ma chai last item pani lincha explicitly vako le

population["California":"New York"]

pd.Series(7, index=[10, 20, 30, 30])

pd.Series({2: "a", 5: "b", 3: "c"})

# user's code overrides the library
pd.Series({2: "a", 5: "b", 3: "c"}, index=[3, 2])

# 2D - dataframes

area_dic = {
    "California": 423,
    "Texas": 695,
    "New York": 141,
    "Florida": 170,
    "Illinois": 150,
}

area = pd.Series(area_dic)
area

states = pd.DataFrame({"population": population, "area": area})

# matrix form ma garo hunthyo so we just did in dataframe for visualization

type(states)

states.index
states.columns

states["area"]

print(population)

pd.DataFrame(population, columns=["population"])
# population series thyo tyo chai column ma rakheko

data = [{"a": i, "b": 2 * i} for i in range(6)]
pd.DataFrame(data)

# table ma kei data chaina vaney chai pandas le NaN rakhdincha

# NaN is not a number

pd.DataFrame([{"a": 0, "b": 1}, {"b": 2, "c": 3}])

# NaN ma kei operation garyo vaney NaN nei aucha so data cleaning ma first step is to check NaN values

# interview question
# data exploration is the starting point
# such as .ndim, .shape, .size, .dtype, .describe, .info

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=["a", "b", "c", "d"])
data

data.keys()  # check what the keys we have

data.values

print(data.items())

list(data.items())

data["e"] = 1.25

data["d"] = 1.25

data = pd.Series(["x", "y", "z"], index=[1, 3, 5])

data[1]

# user defined priority thyo 1 index ma

# loc = location and iloc = implicit location

data.loc[1]

data.iloc[1]

# explicit is better than implicit

states

states["density"] = states["population"] / states["area"]

states
