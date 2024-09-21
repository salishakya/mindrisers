# Activity_Course 4 TikTok project lab

# pandas, numpy, matplotlib.pyplot, seaborn, and scipy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

data = pd.read_csv("../csv/tiktok_dataset.csv")
data.head()

data.describe()
data.columns

# Check for missing values
data.isnull()

# Drop rows with missing values
data.dropna(inplace=True)

# Compute the mean `video_view_count` for each group in `verified_status`
data["verified_status"].unique()

verified_data = data[data["verified_status"] == "verified"]
mean_verified_data = verified_data["video_view_count"].mean()

not_verified_data = data[data["verified_status"] == "not verified"]
mean_not_verified_data = not_verified_data["video_view_count"].mean()

ab_test = stats.ttest_ind(
    verified_data["video_view_count"],
    not_verified_data["video_view_count"],
    equal_var=False,
)

# here p value is 2.6 so we failt to reject the null hypothesis.
