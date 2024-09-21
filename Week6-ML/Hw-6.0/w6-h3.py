import pandas as pd
from scipy import stats

df = pd.read_csv("../csv/waze_dataset.csv")
df.head()
df.describe()
df.isnull()

df.dropna(inplace=True)

map_dictionary = {"Android": 2, "iPhone": 1}

new_df = df

new_df["map_device"] = new_df["device"]

test_df = new_df[["drives", "map_device"]]

test_df["map_device"] = test_df["map_device"].map(map_dictionary)

android_users = test_df[test_df["map_device"] == 2]
iphone_users = test_df[test_df["map_device"] == 1]

ab_test = stats.ttest_ind(
    android_users["drives"], iphone_users["drives"], equal_var=False
)

# here the pvalue is 0.09 so we fail to reject the null hypothesis, hence no difference.
