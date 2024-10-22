# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta

dataset = pd.read_csv("./csv/bank_loan.csv")
dataset.head()

dataset.columns

dataset.drop(columns=["Unnamed: 0"], inplace=True)

# Step 2: Data Scaling
# Scaling the features to make them have zero mean and unit variance.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    dataset[
        [
            "Loan_Amount",
            "Loan_Term",
            "Applicant_Income",
            "Coapplicant_Income",
            "Credit_Score",
            "Property_Area",
            "Employment_Years",
            "Debt_to_Income_Ratio",
            "Collateral_Value",
        ]
    ]
)

# Step 3: Determining Optimal Number of Clusters (Elbow Method)
inertia_values = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)

inertia_values

# Plotting the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia_values, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method to Determine Optimal Number of Clusters")
plt.grid(True)
plt.show()

# Step 4: Choosing the Number of Clusters Based on Silhouette Score
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, labels))

# Plotting Silhouette Scores for Different Values of K
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method to Determine Optimal Number of Clusters")
plt.grid(True)
plt.show()

# From both the Elbow and Silhouette methods, we will pick K=5

# Step 5: Applying KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add the cluster labels to the original dataset
dataset["Cluster"] = cluster_labels

# Add the cluster labels to the original dataset
cluster_names = {
    0: "Low Risk Home Loans",
    1: "High Income Auto Loans",
    2: "Moderate Risk Personal Loans",
    3: "Education Loans with Coapplicant",
    4: "Agricultural Loans with High Collateral",
}
dataset["Cluster"] = pd.Series(cluster_labels).replace(cluster_names)

# Step 6: Visualizing the Clusters (Using PCA to Reduce Dimensions to 2D)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
dataset["PCA1"] = pca_data[:, 0]
dataset["PCA2"] = pca_data[:, 1]

# Plotting the clusters in the 2D PCA space
plt.figure(figsize=(10, 8))
colors = ["red", "blue", "green", "purple", "orange"]
for cluster_name, color in zip(cluster_names.values(), colors):
    cluster_data = dataset[dataset["Cluster"] == cluster_name]
    plt.scatter(
        cluster_data["PCA1"], cluster_data["PCA2"], color=color, label=cluster_name
    )


plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Visualization of Clusters after PCA")
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Evaluating the Model
# Calculating the Silhouette Score for the chosen K
final_silhouette_score = silhouette_score(scaled_data, cluster_labels)
print(f"Silhouette Score for K=5: {final_silhouette_score:.2f}")

# Step 8: Output the Data Segregation Summary Table
cluster_summary = dataset.groupby("Cluster").size().reset_index(name="Count")
print(cluster_summary)

# Step 9: Output the Data Segregation Summary Table
cluster_summary = (
    dataset[["Loan_Type", "Cluster"]]
    .groupby("Cluster")
    .size()
    .reset_index(name="Count")
)
cluster_summary["Default_Loan"] = dataset["Loan_Type"]
print(cluster_summary)

# Creating new features
dataset["Loan_to_Income_Ratio"] = dataset["Loan_Amount"] / (
    dataset["Applicant_Income"] + 1
)
dataset["Collateral_Coverage"] = dataset["Collateral_Value"] / (
    dataset["Loan_Amount"] + 1
)
dataset["Property_to_Income_Ratio"] = dataset["Property_Area"] / (
    dataset["Applicant_Income"] + 1
)

# Creating binned features
bins = [300, 580, 670, 740, 850]
labels = ["Poor", "Average", "Good", "Excellent"]
dataset["Credit_Category"] = pd.cut(dataset["Credit_Score"], bins=bins, labels=labels)

# Log transformation to reduce skewness
dataset["Log_Loan_Amount"] = np.log1p(dataset["Loan_Amount"])

# Adding interaction terms
dataset["Loan_Amount_Interest"] = (
    dataset["Loan_Amount"]
    * dataset["Interest_Rate"].str.rstrip("%").astype("float")
    / 100
)

# Feature extraction from date
dataset["Application_Age_Days"] = (
    datetime.now() - pd.to_datetime(dataset["Application_Date"])
).dt.days

# Display the first few rows of the dataset
print(dataset.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    dataset[
        [
            "Loan_Amount",
            "Loan_Term",
            "Applicant_Income",
            "Coapplicant_Income",
            "Credit_Score",
            "Property_Area",
            "Employment_Years",
            "Debt_to_Income_Ratio",
            "Collateral_Value",
            "Loan_to_Income_Ratio",
            "Collateral_Coverage",
            "Property_to_Income_Ratio",
            "Log_Loan_Amount",
            "Loan_Amount_Interest",
            "Application_Age_Days",
        ]
    ]
)

inertia_values = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertia_values, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method to Determine Optimal Number of Clusters")
plt.grid(True)
plt.show()

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    silhouette_scores.append(silhouette_score(scaled_data, labels))

# Plotting Silhouette Scores for Different Values of K
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method to Determine Optimal Number of Clusters")
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)

# Add the cluster labels to the original dataset
cluster_names = {
    0: "Low Risk Home Loans",
    1: "High Income Auto Loans",
    2: "Moderate Risk Personal Loans",
    3: "Education Loans with Coapplicant",
    4: "Agricultural Loans with High Collateral",
}
dataset["Cluster"] = pd.Series(cluster_labels).replace(cluster_names)

# Step 6: Visualizing the Clusters (Using PCA to Reduce Dimensions to 2D)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
dataset["PCA1"] = pca_data[:, 0]
dataset["PCA2"] = pca_data[:, 1]

plt.figure(figsize=(10, 8))
colors = ["red", "blue", "green", "purple", "orange"]
for cluster_name, color in zip(cluster_names.values(), colors):
    cluster_data = dataset[dataset["Cluster"] == cluster_name]
    plt.scatter(
        cluster_data["PCA1"], cluster_data["PCA2"], color=color, label=cluster_name
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Visualization of Clusters after PCA (with Feature Engineering)")
plt.legend()
plt.grid(True)
plt.show()

final_silhouette_score = silhouette_score(scaled_data, cluster_labels)
print(f"Silhouette Score for K=5: {final_silhouette_score:.2f}")

# Step 8: Output the Data Segregation Summary Table
cluster_summary = (
    dataset[["Loan_Type", "Cluster"]]
    .groupby(["Cluster", "Loan_Type"])
    .size()
    .reset_index(name="Count")
)
print(cluster_summary)

# Display the full dataset with cluster information
print(dataset.head())
