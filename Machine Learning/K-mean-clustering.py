from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("./csv/daraz.csv")

data.columns

data.head()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[["Latitude", "Longitude"]])

data_scaled[0][0]

plt.figure(figsize=(10, 6))
plt.scatter(data_scaled["Longitude"], data_scaled["Latitude"], alpha=0.6)
plt.title("Distributor Locations in Nepal")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()
