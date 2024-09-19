# Linear regression using sklearn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

income_data = pd.read_csv("./income.data.csv")

income_data.dropna()

income = income_data["income"]
happiness = income_data["happiness"]

plt.figure(figsize=(8, 6))
plt.scatter(income, happiness, color="blue")
plt.title("Income relative to happiness")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    income, happiness, test_size=0.2, random_state=42
)

# Print shapes for debugging
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the linear regression model
model = LinearRegression()

X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)

# Train the model
model.fit(X_train, y_train)

X_test = pd.DataFrame(X_test)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model (optional, but useful for understanding performance)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="blue")
plt.plot(X_test, y_pred, color="red", linewidth=2)
plt.title("Income-Happiness Prediction")
plt.xlabel("Income")
plt.ylabel("Happiness")
plt.grid(True)
plt.show()

# Predict exam score for a new study hour
new_income = np.array([[9]])
predicted_score = model.predict(new_income)
print(predicted_score)
