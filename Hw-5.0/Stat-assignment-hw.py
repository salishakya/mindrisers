import pandas as pd


data = pd.read_csv("ecommerce_data.csv")
data.columns

data["Purchase_Amount"].mean()
data["Purchase_Amount"].median()
data["Purchase_Amount"].mode()

data["Purchase_Amount"].head(5)

data["Purchase_Amount"].std()
data["Purchase_Amount"].var()

data["Age"].mean()
data.head()

# quartiles of Age
data["Age"].quantile(0.25)
data["Age"].quantile(0.5)
data["Age"].quantile(0.75)

data_range = data["Age"].max() - data["Age"].min()
data_range

data["Purchase_Category"].head()

# counting the categories

category_count = data["Purchase_Category"].value_counts()
print(category_count)

data.shape

data.boxplot(column="Purchase_Amount")
data.hist(column="Purchase_Amount")

# Simple Random Sampling: Select 10% of the customers
simple_random_sample = data.sample(frac=0.1, random_state=42)

# Stratified Sampling: : Select a sample based on Purchase_Category, ensuring each category is proportionally represented.
stratified_sample = data.groupby("Purchase_Category", group_keys=False).apply(
    lambda x: x.sample(frac=0.01, random_state=42)
)

data.index

len(data)


systematic_sample_list = []
for i in range(len(data)):
    systematic_sample_list.append(data.iloc[i])
    i = i + 10
systematic_sample_df = pd.DataFrame(systematic_sample_list)
systematic_sample_df

# The issue code is that the line i = i + 10 inside the loop doesn’t affect the range(len(data)) iteration. It only modifies the local variable i, not the loop’s progression.

systematic_sample_list = []
for i in range(0, len(data), 10):  # Step by 10
    systematic_sample_list.append(data.iloc[i])

systematic_sample_df = pd.DataFrame(systematic_sample_list)
systematic_sample_df

data.columns

cluster_sample = data.groupby("Country")
