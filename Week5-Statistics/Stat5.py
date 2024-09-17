import pandas as pd
from scipy import stats

education_districtwise = pd.read_csv("../Week4/education_districtwise.csv")
education_districtwise = education_districtwise.dropna()

education_districtwise.columns

education_districtwise["STATNAME"].unique()

punjab = education_districtwise[education_districtwise["STATNAME"] == "PUNJAB"]
bihar = education_districtwise[education_districtwise["STATNAME"] == "BIHAR"]
punjab.head()

sample_punjab = punjab.sample(n=20, replace=True, random_state=42)
sample_bihar = bihar.sample(n=20, replace=True, random_state=42)

sample_punjab["OVERALL_LI"].mean()
sample_bihar["OVERALL_LI"].mean()

# Hypothesis Testing

# Null hypothesis: There is not difference in literacy rate
# Alternate hypothesis: There is difference in hypotesis
# significance level: 5% error vaye null hypothesis manne

stats.ttest_ind(
    a=sample_bihar["OVERALL_LI"], b=sample_punjab["OVERALL_LI"], equal_var=False
)

# T value is -5.01579
# p value = 1.2648

stats.ttest_ind(
    a=sample_punjab["OVERALL_LI"], b=sample_bihar["OVERALL_LI"], equal_var=False
)
