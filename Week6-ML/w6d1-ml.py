import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulated data for study hours and exam scores
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Input feature (X)
exam_scores = np.array([45, 55, 60, 65, 70, 75, 80, 85, 90, 95])  # Target variable (y)

# Visualize the data
plt.figure(figsize=(8, 6))
plt.scatter(study_hours, exam_scores, color="blue")
plt.title("Exam Score vs Study Hours")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    study_hours.reshape(-1, 1), exam_scores, test_size=0.2, random_state=42
)

# Print shapes for debugging
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

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
plt.title("Exam Score Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.grid(True)
plt.show()

# Predict exam score for a new study hour
new_study_hour = np.array([[9]])  # Predict for 7 hours of study

predicted_score = model.predict(new_study_hour)
print(
    f"Predicted exam score for {new_study_hour[0][0]} hours of study: {predicted_score[0]}"
)
