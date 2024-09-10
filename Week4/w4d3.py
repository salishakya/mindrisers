import pandas as pd

education_districtwise = pd.read_csv("education_districtwise.csv")

education_districtwise.head()

education_districtwise.describe()

education_districtwise.columns

"OVERALL_LI" in education_districtwise.columns

education_districtwise["OVERALL_LI"].describe()

education_districtwise["STATNAME"].describe()

range_overall_li = (
    education_districtwise["OVERALL_LI"].max()
    - education_districtwise["OVERALL_LI"].min()
)
range_overall_li
