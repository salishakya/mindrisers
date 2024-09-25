import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("./csv/data.csv")
data.head()
data.isnull().sum()

data.columns

sns.countplot(data=data, x="Survived")
plt.title("Survival count")
plt.show()

sns.histplot(data["Age"], kde=True)
plt.title("Age Distrubution")
plt.show()

sns.boxplot(x="Survived", y="Age", data=data)
plt.title("Survival by Age")
plt.show()
