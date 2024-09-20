# Activity_ Course 4 Automatidata project lab

from numpy import dtype
import pandas as pd
from scipy import stats

taxi_data = pd.read_csv("../csv/2017_Yellow_Taxi_Trip_Data.csv", index_col=0)
taxi_data.describe()
taxi_data.head()
taxi_data.columns

# In the dataset, payment_type is encoded in integers: * 1: Credit card * 2: Cash * 3: No
# charge * 4: Dispute * 5: Unknown

print(taxi_data["payment_type"].unique())

# Hypothesis testing

# H0: There is no difference in the average fare amount between customers who use credit cards and
# customers who use cash.
# HA: There is a difference in the average fare amount between customers who use credit cards and
# customers who use cash.

dtype(taxi_data["payment_type"])

customers_cc = taxi_data[taxi_data["payment_type"] == 1]
customers_cc.head()

customers_cash = taxi_data[taxi_data["payment_type"] == 2]
customers_cash.head()

sample_customer_cc = customers_cc.sample(n=50, replace=True, random_state=42)
sample_customer_cash = customers_cash.sample(n=50, replace=True, random_state=42)

sample_customer_cc.head()
sample_customer_cash.head()

ab_test = stats.ttest_ind(
    a=sample_customer_cc["fare_amount"],
    b=sample_customer_cash["fare_amount"],
    equal_var=False,
)

sample_customer_cc.columns

# here the p value is 0.80 i.e. more than 0.05 significance value so we fail to reject the null hypothesis.
