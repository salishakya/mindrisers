# Activity: Explore hypothesis testing

import numpy as np
import pandas as pd
from scipy import stats

aqi = pd.read_csv("c4_epa_air_quality.csv")

aqi.describe()
aqi.columns

aqi["county_name"].unique()

la = aqi[aqi["county_name"] == "Los Angeles"]
not_la = aqi[aqi["county_name"] != "Los Angeles"]

la.shape
not_la.shape

sample_la = la.sample(n=10, replace=False, random_state=42)
sample_not_la = not_la.sample(n=10, replace=False, random_state=42)

sample_la.columns

stats.ttest_ind(a=sample_la["aqi"], b=sample_not_la["aqi"], equal_var=False)

# the p value is 0.0887 so it's 8.87% which means that, Since the p-value (0.0887) is greater than the significance level (0.05):

# You fail to reject the null hypothesis.
# This means there isn't enough evidence to conclude that the observed result is statistically significant at the 5% significance level.
# In simple terms, the data doesn't provide strong enough evidence to reject the null hypothesis, meaning the results could have occurred by random chance under the assumption that the null hypothesis is true.

# you fail to reject the null hypothesis, so null hypothesis is true.

# #Hypothesis 2: With limited resources, ROA has to choose between New York
# and Ohio for their next regional office. Does New York have a lower AQI than
# Ohio?

aqi["state_name"].unique()

ny = aqi[aqi["state_name"] == "New York"]
ohio = aqi[aqi["state_name"] == "Ohio"]

ny.shape
ohio.shape

sample_ny = ny.sample(n=50, replace=True, random_state=42)
sample_ohio = ohio.sample(n=50, replace=True, random_state=42)

stats.ttest_ind(a=sample_ny["aqi"], b=sample_ohio["aqi"], equal_var=False)

# since the pvalue is less than 5% so we can accept the null hypothesis that The mean AQI of New York is greater than or equal to that of Ohio.

# Hypothesis 3: A new policy will affect those states with a mean AQI of 10 or
# greater. Will Michigan be affected by this new policy?

# H0: The mean AQI of Michigan is less than or equal to 10.
# HA: The mean AQI of Michigan is greater than 10.

mich = aqi[aqi["state_name"] == "Michigan"]

sample_mich = mich.sample(n=50, replace=True)

# check documentation for 1sample. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html

stats.ttest_1samp(sample_mich["aqi"], 10, alternative="greater")

# here p value is 99.98% so we failed to reject null hypothesis which means that the mean AQI of Michigan is NOT less than or equal to 10.
