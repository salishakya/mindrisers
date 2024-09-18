# linear regression on exam score and income-happiness using scipy

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# Simulated data for study hours and exam scores
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Input feature (X)
exam_scores = np.array([45, 55, 60, 65, 70, 75, 80, 85, 90, 95])  # Target variable (y)

model = stats.linregress(study_hours, exam_scores)

m = model.slope
c = model.intercept

# linear equation
y = m * study_hours + c

# Plotting the data points
plt.scatter(study_hours, exam_scores, color="blue", label="Actual Data")

# Plotting the best fit line
plt.plot(study_hours, y, color="red", label=f"Best fit line")

# Adding the legend
plt.legend()

plt.show()

income_data = pd.read_csv("./income.data.csv")
income_data.head()

plt.plot(income_data["income"], income_data["happiness"])

plt.scatter(income_data["income"], income_data["happiness"])

income = income_data["income"]
happiness = income_data["happiness"]

model = stats.linregress(income, happiness)
model

y_happiness = model.slope * income + model.intercept

# Plotting the data points
plt.scatter(income, happiness, color="blue", label="Actual Data")

# Plotting the best fit line
plt.plot(income, y_happiness, color="red", label="Best fit line")

# Adding the legend
plt.legend()

plt.show()
