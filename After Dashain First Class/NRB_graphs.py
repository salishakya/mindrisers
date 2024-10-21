import pandas as pd
import numpy as np
import warnings

from streamlit import columns

warnings.filterwarnings("ignore")

xls = pd.ExcelFile("./xlsx/Shrawan_2081_Publish-1.xlsx")

dataframes = []

for i, sheet_name in enumerate(xls.sheet_names):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    dataframes.append(df)

for index, dataframe in enumerate(dataframes, start=1):
    exec(f"data{index} = dataframe")

# data 4 means 4th sheet
data4 = data4.drop([0, 1, 3]).reset_index(drop=True)  # type: ignore
data4.head()

data4.drop(columns=["Unnamed: 0", "Unnamed: 1"], inplace=True)

data4_1 = data4.iloc[:12]
data4_1.columns = data4_1.iloc[0]
data4_1 = data4_1.drop(0).reset_index(drop=True)
data4_1.iloc[:11]
data4_1.rename(columns={np.nan: "Credit, Deposit Ratios (%)"}, inplace=True)
data4_1

import matplotlib.pyplot as plt

# Assuming 'data4_1' is a pandas DataFrame
# Extracting categories (first column) and the class values (Class A, B, C, Overall)
categories = data4_1.iloc[:, 0]  # First column for category labels
class_a_values = data4_1['Class "A"']
class_b_values = data4_1['Class "B"']
class_c_values = data4_1['Class "C"']
overall_values = data4_1["Overall"]

# Create a grouped bar plot
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.2
x_positions = np.arange(len(categories))

# Plotting bars for each class
ax.barh(
    x_positions - bar_width, class_a_values, bar_width, label="Class A", color="skyblue"
)
ax.barh(x_positions, class_b_values, bar_width, label="Class B", color="lightgreen")
ax.barh(
    x_positions + bar_width, class_c_values, bar_width, label="Class C", color="salmon"
)
ax.barh(
    x_positions + 2 * bar_width,
    overall_values,
    bar_width,
    label="Overall",
    color="gray",
)

# Adding labels and title
ax.set_yticks(x_positions)
ax.set_yticklabels(categories)
ax.set_xlabel("Percentage (%)")
ax.set_title("Credit and Deposit Ratios by Class")

# Adding legend
ax.legend()

plt.tight_layout()
plt.show()
