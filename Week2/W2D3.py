import numpy as np
import pandas as pd

data1 = pd.read_csv("Seattle2014.csv")

rainfall = data1["PRCP"].values
rainfall

inches = rainfall / 264  # converting to inches
inches

import matplotlib.pyplot as plt  # noqa: E402
import seaborn  # noqa: E402

seaborn.set_theme()

plt.hist(inches, bins=30)

np.count_nonzero(inches > 0.5)

np.sum((inches > 0.5) & (inches < 1.0))  # true and false count garcha

# mask of all rainy days
rainy = inches > 0
rainy[:20]
np.count_nonzero(rainy)

summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)

summer

# summer inches ko see at home

rand = np.random.RandomState(42)  # memory location ma gayera rakhcha just like seed
x = rand.randint(100, size=10)
x

# access elements

[x[1], x[3]]

indices = [1, 3, 0]
x[indices]

indices = np.array([[2, 7], [8, 4]])
indices

x[indices]

# fancy indexing vancha eslai

x = np.array([1, 3, 2, 4, 5])
np.sort(x)

# x will be unchanged

x.sort()
x
# this now changes the value, np le chai chudaina but python ko x.sort le chai chuncha

x = np.array([2, 3, -2, 7, 0])
np.argsort(x)

# argument sort figureouts the position of the array

# partial sorts

x = np.array([7, 2, 3, 4, 6, 8, 9])
print(x)
np.partition(x, 3)

x = np.array([1000, 2, 3, 4, 6, 8, 9])
print(x)
np.partition(x, 3)


x = np.array([1000, 2, 3, 4, 10, 8, 9])
print(x)
np.partition(x, 3)

x = np.array([1000, 2, 4, 1, 10, 8, 9])
print(x)
np.partition(x, 3)

name = ["Alice", "Bob", "Cathy", "Steve"]
age = [35, 42, 25, 15]
weight = [60.3, 90.2, 75.1, 51.0]

data = np.zeros(
    4, dtype={"names": ("name", "age", "weight"), "formats": ("U10", "i4", "f8")}
)
data.dtype

data["name"]

data["name"] = name
data["age"] = age
data["weight"] = weight
data

# structure banako, relation banako

data["name"]

data[0]

data[0][name]

data[0]["name"]

data[0]["name"] = "Micheal"
data[0]

data[data["age"] < 30]["name"]
# inline gareko

np.sum((inches > 0.5) & (inches < 1.0))  # true and false count garcha
