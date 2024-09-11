# binomial, possion, normal distrubution, z score for outliers

import pandas as pd
from scipy import stats

education_districtwise = pd.read_csv("../week4/education_districtwise.csv")
# removing NaN values
education_districtwise = education_districtwise.dropna()

education_districtwise.head()

education_districtwise["OVERALL_LI"].hist()

mean_overall_li = education_districtwise["OVERALL_LI"].mean()
mean_overall_li

std_overall_li = education_districtwise["OVERALL_LI"].std()
std_overall_li

lower_limit = mean_overall_li - 1 * std_overall_li
upper_limit = mean_overall_li + 1 * std_overall_li
(
    (education_districtwise["OVERALL_LI"] >= lower_limit)
    & (education_districtwise["OVERALL_LI"] <= upper_limit)
).mean()

lower_limit = mean_overall_li - 2 * std_overall_li
upper_limit = mean_overall_li + 2 * std_overall_li
(
    (education_districtwise["OVERALL_LI"] >= lower_limit)
    & (education_districtwise["OVERALL_LI"] <= upper_limit)
).mean()

lower_limit = mean_overall_li - 3 * std_overall_li
upper_limit = mean_overall_li + 3 * std_overall_li
(
    (education_districtwise["OVERALL_LI"] >= lower_limit)
    & (education_districtwise["OVERALL_LI"] <= upper_limit)
).mean()

# z score is basically the standard deviation value to check outliers, so if it's less than -3 and more than +3 SD then they are outliers
education_districtwise["Z_SCORE"] = stats.zscore(education_districtwise["OVERALL_LI"])
education_districtwise

education_districtwise[
    (education_districtwise["Z_SCORE"] > 3) | (education_districtwise["Z_SCORE"] < -3)
]

# Here Dantewada and alirajpur are both outlieres with too low of a literacy rate
