import pandas as pd
from scipy import stats

data = pd.read_csv("modified_c4_epa_air_quality.csv")
# removing NaN values
data = data.dropna()

data.head()

data["aqi_log"].hist()

mean_aqi_log = data["aqi_log"].mean()
mean_aqi_log

std_aqi_log = data["aqi_log"].std()
std_aqi_log

lower_limit = mean_aqi_log - 1 * std_aqi_log
upper_limit = mean_aqi_log + 1 * std_aqi_log

print("1SD lower limit: ", lower_limit)
print("1SD upper limit:", upper_limit)

print(
    "Should be around 64% to be normal distribution: ",
    ((data["aqi_log"] >= lower_limit) & (data["aqi_log"] <= upper_limit)).mean(),
)

lower_limit = mean_aqi_log - 2 * std_aqi_log
upper_limit = mean_aqi_log + 2 * std_aqi_log

print("2SD lower limit: ", lower_limit)
print("2SD upper limit:", upper_limit)

print(
    "Should be around 92% to be normal distribution: ",
    ((data["aqi_log"] >= lower_limit) & (data["aqi_log"] <= upper_limit)).mean(),
)

lower_limit = mean_aqi_log - 3 * std_aqi_log
upper_limit = mean_aqi_log + 3 * std_aqi_log

print("3SD lower limit: ", lower_limit)
print("3SD upper limit:", upper_limit)

print(
    "Should be around 96% to be normal distribution: ",
    ((data["aqi_log"] >= lower_limit) & (data["aqi_log"] <= upper_limit)).mean(),
)

# z score is basically the standard deviation value to check outliers, so if it's less than -3 and more than +3 SD then they are outliers
data["Z_SCORE"] = stats.zscore(data["aqi_log"])
data.head()

data[(data["Z_SCORE"] > 3) | (data["Z_SCORE"] < -3)]

# Here Arizona is the outliere with too high of api log
