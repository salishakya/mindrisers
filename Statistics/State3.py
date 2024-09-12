import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

education_districtwise = pd.read_csv("../Week4/education_districtwise.csv")
education_districtwise = education_districtwise.dropna()

sampled_data = education_districtwise.sample(50, replace=True, random_state=31208)
sampled_data

# replace true le duplicate data ni rakhdincha

sampled_data_no_duplicates = sampled_data["DISTNAME"].duplicated(keep=False)

sampled_data = education_districtwise.sample(50, replace=False, random_state=31208)
sampled_data

sampled_data.columns

estimate1 = sampled_data["OVERALL_LI"].mean()
estimate1

estimate2 = (
    education_districtwise["OVERALL_LI"]
    .sample(50, replace=False, random_state=1)
    .mean()
)
estimate2

estimate_list = []
for i in range(10000):
    estimate_list.append(
        education_districtwise["OVERALL_LI"].sample(n=50, replace=True).mean()
    )
estimate_df = pd.DataFrame(data={"estimate": estimate_list})
estimate_df.head()
estimate_df.mean()
