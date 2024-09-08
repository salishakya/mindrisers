import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seaborn import set_theme

plt.plot([1, 2, 3, 4])
plt.xlabel("number")
plt.ylabel("square of number")
plt.title("Square")

plt.plot([1, 2, 3, 4], [2, 3, 4, 5], "bo")
plt.axis((0, 6, 0, 20))
plt.show()

# axis le chai range of the axis dincha

t = np.arange(0, 5, 0.2)
plt.plot(t, t, "r--", t, t**2, "bs", t, t**3, "g^")
plt.show

# this is all from the official documentation of matplot

data = {"a": np.arange(50), "c": np.random.randint(0, 50, 50), "d": np.random.randn(50)}
data["b"] = data["a"] + 10 * np.random.randn(50)
data["d"] = np.abs(data["d"]) * 100

print(data)
plt.scatter("a", "b", c="c", s="d", data=data)
plt.xlabel("entry a")
plt.ylabel("entry b")
plt.title("scatter plot")
plt.show()

names = ["group a", "group b", "group c"]
values = ["1", "2", "3"]

plt.figure(figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle("categotial plotting")
plt.show()

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

plt.hist(x, bins=50, density=True, facecolor="r", alpha=1)

plt.xlabel("Smarts")
plt.ylabel("Probability")
plt.title("Histogram of IQ")
plt.text(60, 0.25, r"$mu=100,\\signma=15$")
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

data = {
    "Barton LLC": 109438.50,
    "Frami, Hills and Schmidt": 103569.59,
    "Fritsch, Russel and Anderson": 112214.71,
    "Jerde-Hilpert": 112591.43,
    "Keeling LLC": 100934.30,
    "Koepp Ltd": 103660.54,
    "Kulas Inc": 137351.96,
    "Trantow-Barrows": 123381.38,
    "White-Trantow": 135841.99,
    "Will LLC": 104437.60,
}
group_data = list(data.values())
group_names = list(data.keys())
group_mean = np.mean(group_data)

fig, ax = plt.subplots()
ax.barh(group_names, group_data)

fig, ax = plt.subplots()
ax.bar(group_names, group_data)

plt.style.available

plt.style.use("classic")

fig, ax = plt.subplots()
ax.barh(group_names, group_data)

plt.style.use("tableau-colorblind10")

fig, ax = plt.subplots()
ax.barh(group_names, group_data)
labels = ax.get_xticklabels()
