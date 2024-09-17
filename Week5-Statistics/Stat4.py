from matplotlib import scale
import numpy as np
import pandas as pd
from scipy import stats

education_districtwise = pd.read_csv("../Week4/education_districtwise.csv")
education_districtwise = education_districtwise.dropna()

sampled_data = education_districtwise.sample(n=50, replace=True, random_state=31208)
sampled_data

# construct a 95% confidence interval

sample_mean = sampled_data["OVERALL_LI"].mean()

estimated_sample_error = sampled_data["OVERALL_LI"].std() / np.sqrt(
    sampled_data.shape[0]
)

# confidence interval of 95%

stats.norm.interval(0.95, loc=sample_mean, scale=estimated_sample_error)

lower_bound, upper_bound = stats.norm.ppf(
    [0.025, 0.975], loc=sample_mean, scale=estimated_sample_error
)

stats.norm.interval(0.99, loc=sample_mean, scale=estimated_sample_error)

aqi = pd.read_csv("./c4_epa_air_quality.csv")

aqi.describe()
aqi.shape

aqi.columns

np.unique(aqi["state_name"])

# we just need some states

rre_states = ["California", "Florida", "Michigan", "Ohio", "Pennsylvania", "Texas"]

aqi_rre = aqi[aqi["state_name"].isin(rre_states)]

aqi_rre.groupby(["state_name"]).agg({"aqi": "mean", "state_name": "count"})

import seaborn as sns

sns.boxplot(x=aqi_rre["state_name"], y=aqi_rre["aqi"])

# cal and mich have higher than 10 aqi

aqi_ca = aqi[aqi["state_name"] == "California"]

ca_aqi_mean = aqi_ca["aqi"].mean()
ca_aqi_mean

standard_error = ca_aqi_mean / np.sqrt(aqi_ca.shape[0])
stats.norm.interval(0.95, loc=ca_aqi_mean, scale=standard_error)
