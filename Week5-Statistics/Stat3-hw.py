import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epa_data = pd.read_csv("c4_epa_air_quality.csv")
epa_data.head(10)
epa_data.describe()

population_mean = epa_data["aqi"].mean()

sampled_data = epa_data.sample(50, replace=True, random_state=42)
sampled_data.head(10)

sample_mean = sampled_data["aqi"].mean()
sample_mean

estimate_list = []
for i in range(10000):
    estimate_list.append(epa_data["aqi"].sample(50, replace=False).mean())
estimate_list

estimate_df = pd.DataFrame(data={"estimate": estimate_list})

mean_sample_means = estimate_df["estimate"].mean()

# Calculate the sample standard deviation
sample_std = sampled_data["aqi"].std()

# standard error of mean
standard_error = sample_std / np.sqrt(50)
print("Standard Error of the Mean AQI:", standard_error)

# actual mean is 6.75.... sampled mean is 6.76.... the standard error is low = 0.74 so high precision :)

# standard error can be around 10% of actual mean - the rule of thumb. If more than 10% it's considered low precision

# Plot the histogram of the 10,000 sample means
plt.hist(estimate_df["estimate"], bins=30, edgecolor="black", alpha=0.7)

# Add a vertical line for the sample mean
plt.axvline(
    sample_mean,
    color="red",
    linestyle="dashed",
    linewidth=2,
    label=f"Single Sample Mean: {sample_mean:.2f}",
)

# Add a vertical line for the mean of the sample means
plt.axvline(
    mean_sample_means,
    color="blue",
    linestyle="solid",
    linewidth=2,
    label=f"Mean of Sample Means: {mean_sample_means:.2f}",
)

# Add a vertical line for the population mean
plt.axvline(
    population_mean,
    color="green",
    linestyle="dotted",
    linewidth=2,
    label=f"Population Mean: {population_mean:.2f}",
)

# Add titles and labels
plt.title("Distribution of 10,000 Sample Means (n=50)")
plt.xlabel("AQI Sample Means")
plt.ylabel("Frequency")

# Display the legend
plt.legend()

# Show the plot
plt.show()
