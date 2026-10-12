# loading the packages and dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 50)
print("CUSTOMER SEGMENTATION PROJECT")
print("=" * 50)

# Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Display the first five records
print("\nFirst Five Records:")
print(df.head())

# Display the number of rows and columns
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Check data types
print("\nData Types:")
print(df.dtypes)

print("\nModule 1 completed successfully!")


# Data Cleaning and Preparation

print("\n" + "=" * 50)
print("MODULE 2: DATA CLEANING AND PREPARATION")
print("=" * 50)

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records Before Cleaning:")
print(df.duplicated().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Check for missing values after cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Remove CustomerID because it is only an identifier
df = df.drop(columns=["CustomerID"])

# Display the cleaned dataset
print("\nFirst Five Records After Cleaning:")
print(df.head())

# Display the updated dataset shape
print("\nDataset Shape After Cleaning:")
print(df.shape)

# Display the final column names
print("\nColumns After Cleaning:")
print(df.columns.tolist())

print("\nModule 2 completed successfully!")

# Exploratory Data Analysis (EDA)

print("\n" + "=" * 50)
print("MODULE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# 1. Gender distribution
print("\nCustomer Count by Gender:")
print(df["Gender"].value_counts())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Gender")
plt.title("Customer Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 2. Age distribution
print("\nAge Statistics:")
print(df["Age"].describe())

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=15, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 3. Annual income distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Annual Income (k$)", bins=15, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 4. Spending score distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Spending Score (1-100)", bins=15, kde=True)
plt.title("Customer Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

# 5. Annual income vs spending score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender"
)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.show()

print("\nModule 3 completed successfully!")

# Analyze Relationships Between Features

print("\n" + "=" * 50)
print("MODULE 4: FEATURE RELATIONSHIP ANALYSIS")
print("=" * 50)

# 1. Age vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Age",
    y="Spending Score (1-100)",
    hue="Gender"
)
plt.title("Age vs Spending Score")
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.show()

# 2. Age vs Annual Income
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Age",
    y="Annual Income (k$)",
    hue="Gender"
)
plt.title("Age vs Annual Income")
plt.xlabel("Age")
plt.ylabel("Annual Income (thousands of dollars)")
plt.tight_layout()
plt.show()

# 3. Annual Income vs Spending Score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender"
)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score")
plt.tight_layout()
plt.show()

print("\nModule 4 completed successfully!")

# Select Features for Clustering

print("\n" + "=" * 50)
print("MODULE 5: FEATURE SELECTION")
print("=" * 50)

# Select the features used to group customers
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

print("\nSelected Features:")
print(X.head())

print("\nFeature Dataset Shape:")
print(X.shape)

print("\nFeature Names:")
print(X.columns.tolist())

print("\nModule 5 completed successfully!")

# Feature Scaling

from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 50)
print("MODULE 6: FEATURE SCALING")
print("=" * 50)

# Create the scaler
scaler = StandardScaler()

# Scale the selected features
X_scaled = scaler.fit_transform(X)

# Convert scaled values back into a DataFrame for easy viewing
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nFirst Five Rows Before Scaling:")
print(X.head())

print("\nFirst Five Rows After Scaling:")
print(X_scaled.head())

print("\nScaled Dataset Shape:")
print(X_scaled.shape)

print("\nMean of Scaled Features:")
print(X_scaled.mean().round(2))

print("\nStandard Deviation of Scaled Features:")
print(X_scaled.std().round(2))

print("\nModule 6 completed successfully!")

# Module 7: Find Optimal Number of Clusters

from sklearn.cluster import KMeans

print("\n" + "=" * 50)
print("MODULE 7: ELBOW METHOD")
print("=" * 50)

# Store the WCSS values
wcss = []

# Test cluster counts from 1 to 10
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

    print(f"Clusters: {i}, WCSS: {kmeans.inertia_:.2f}")

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.xticks(range(1, 11))
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nModule 7 completed successfully!")

# Train K-Means Clustering Model

from sklearn.cluster import KMeans

print("\n" + "=" * 50)
print("MODULE 8: K-MEANS MODEL TRAINING")
print("=" * 50)

# Create the K-Means model
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

# Train the model and assign each customer to a cluster
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Display the cluster assigned to each customer
print("\nFirst 10 Customers with Cluster Assignments:")
print(df.head(10))

# Display the number of customers in each cluster
print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# Display the cluster centers
print("\nCluster Centers (Scaled Values):")
print(kmeans.cluster_centers_)

print("\nModel Inertia (WCSS):")
print(round(kmeans.inertia_, 2))

print("\nModule 8 completed successfully!")

# Visualize Customer Segments

print("\n" + "=" * 50)
print("MODULE 9: CUSTOMER SEGMENT VISUALIZATION")
print("=" * 50)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=100
)

# Convert cluster centers back to the original feature scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=250,
    color="black",
    label="Cluster Centers"
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score")
plt.legend()
plt.tight_layout()
plt.show()

print("\nModule 9 completed successfully!")

# Analyze Customer Segments

print("\n" + "=" * 50)
print("MODULE 10: CUSTOMER SEGMENT ANALYSIS")
print("=" * 50)

# Calculate average values for each cluster
cluster_summary = df.groupby("Cluster").agg(
    Customer_Count=("Cluster", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean")
).round(2)

print("\nCustomer Segment Summary:")
print(cluster_summary)

# Display the cluster with the highest average spending score
highest_spending_cluster = cluster_summary[
    "Average_Spending_Score"
].idxmax()

print(
    f"\nCluster {highest_spending_cluster} "
    "has the highest average spending score."
)

# Display the cluster with the highest average income
highest_income_cluster = cluster_summary[
    "Average_Income"
].idxmax()

print(
    f"Cluster {highest_income_cluster} "
    "has the highest average annual income."
)

print("\nModule 10 completed successfully!")

# Business Insights and Recommendations

print("\n" + "=" * 50)
print("MODULE 11: BUSINESS INSIGHTS")
print("=" * 50)

for cluster_id, row in cluster_summary.iterrows():

    income = row["Average_Income"]
    spending = row["Average_Spending_Score"]

    print(f"\nCustomer Segment: Cluster {cluster_id}")
    print(f"Number of Customers: {int(row['Customer_Count'])}")
    print(f"Average Annual Income: ${income:.2f}k")
    print(f"Average Spending Score: {spending:.2f}")

    if income >= 70 and spending >= 60:
        print("Segment Type: High-Income, High-Spending")
        print("Recommendation: Offer loyalty rewards and premium products.")

    elif income >= 70 and spending < 40:
        print("Segment Type: High-Income, Low-Spending")
        print("Recommendation: Test personalized offers to encourage engagement.")

    elif income < 40 and spending >= 60:
        print("Segment Type: Lower-Income, High-Spending")
        print("Recommendation: Promote popular products and suitable offers.")

    elif income < 40 and spending < 40:
        print("Segment Type: Lower-Income, Low-Spending")
        print("Recommendation: Test budget-friendly promotions.")

    else:
        print("Segment Type: Moderate-Income or Moderate-Spending")
        print("Recommendation: Use targeted campaigns based on customer preferences.")

print("\nModule 11 completed successfully!")

# Save the Trained Model

import os
import joblib

print("\n" + "=" * 50)
print("MODULE 12: SAVE THE TRAINED MODEL")
print("=" * 50)

# Create a folder to store the saved files
os.makedirs("saved_model", exist_ok=True)

# Save the trained K-Means model
joblib.dump(kmeans, "saved_model/kmeans_model.pkl")

# Save the scaler used during training
joblib.dump(scaler, "saved_model/scaler.pkl")

# Save the cluster summary
cluster_summary.to_csv("saved_model/cluster_summary.csv")

print("\nK-Means model saved successfully!")
print("Scaler saved successfully!")
print("Cluster summary saved successfully!")

print("\nSaved Files:")
print(os.listdir("saved_model"))

print("\nModule 12 completed successfully!")
# Module 13: Test the Saved Model

import joblib

print("\n" + "=" * 50)
print("MODULE 13: TEST THE SAVED MODEL")
print("=" * 50)

# Load the saved model and scaler
loaded_model = joblib.load("saved_model/kmeans_model.pkl")
loaded_scaler = joblib.load("saved_model/scaler.pkl")

# Create sample customer data
new_customer = pd.DataFrame({
    "Annual Income (k$)": [85],
    "Spending Score (1-100)": [80]
})

# Scale the customer data while preserving feature names
scaled_customer = pd.DataFrame(
    loaded_scaler.transform(new_customer),
    columns=new_customer.columns
)

# Predict the customer cluster
predicted_cluster = loaded_model.predict(scaled_customer)[0]

print("\nNew Customer Details:")
print(new_customer)

print(f"\nPredicted Customer Cluster: {predicted_cluster}")

print("\nModule 13 completed successfully!")


# Export Customer Segments

print("\n" + "=" * 50)
print("MODULE 14: EXPORT CUSTOMER SEGMENTS")
print("=" * 50)

# Export customer records with their cluster assignments
df.to_csv("customer_segments.csv", index=False)

print("\nCustomer segments exported successfully!")
print("File: customer_segments.csv")

print("\nFirst Five Records:")
print(df.head())

print("\nTotal Customers Exported:", len(df))

print("\nModule 14 completed successfully!")

# Evaluate Clustering Quality

from sklearn.metrics import silhouette_score

print("\n" + "=" * 50)
print("MODULE 15: CLUSTERING EVALUATION")
print("=" * 50)

# Calculate the Silhouette Score
score = silhouette_score(X_scaled, df["Cluster"])

print(f"\nSilhouette Score: {score:.4f}")

if score > 0.5:
    print("Interpretation: The clusters are reasonably well separated.")
elif score > 0.25:
    print("Interpretation: The clusters have moderate separation.")
else:
    print("Interpretation: The clusters overlap considerably.")

print("\nModule 15 completed successfully!")

