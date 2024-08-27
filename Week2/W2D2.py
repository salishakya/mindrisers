# step 1: Loading/ Importing library

import numpy as np
import pandas as pd

# step 2: Reading file

data = pd.read_csv("president_heights.csv")

data[:5]

data

data.shape

data.head()
data.tail()

data.head(15)
data.tail(7)

# np only deals with array

data.columns

heights = np.array(data["height(cm)"])

type(heights)

print("Mean: ", heights.mean())
print("Max value: ", heights.max())
print("Min: ", heights.min())
print("SD:", heights.std())

import matplotlib.pyplot as plt  # noqa: E402
import seaborn  # noqa: E402

seaborn.set_theme()

plt.hist(heights)
plt.title("Height of US presidents")  # title of the graph
plt.xlabel("height(cm)")
plt.ylabel("number")

plt.show()

plt.scatter(heights)

names = np.array(data["name"])

plt.scatter(names, heights)
plt.scatter(heights, names)

# broadcasting: operations on different array sizes

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

x + 3

# added 3 to all elements of x here

md_arr = np.ones((3, 3))

md_arr + 3

data1 = pd.read_csv("Seattle2014.csv")

data1.shape

data1.head()

rainfall = data1["PRCP"].values
rainfall

print(rainfall.min())
print(rainfall.max())

inches = rainfall / 264  # converting to inches
inches

plt.hist(inches, bins=30)

# bins le kati ota banauni vanera define garcha, kati ota banauni

plt.hist(inches, bins=5)
