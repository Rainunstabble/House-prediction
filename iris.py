
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("./IRIS.csv")

# Explore dataset
print(df.head())
print(df.info())
print(df.describe())

print("Shape =", df.shape)

# Select features
X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train K-Means model
model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

model.fit(X_scaled)

# Store cluster labels
df["Clusters"] = model.labels_

# Test model on a new flower
new_flower = pd.DataFrame([{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}])

# Scale new flower using the same scaler
new_flower_scaled = scaler.transform(new_flower)

# Predict cluster
prediction = model.predict(new_flower_scaled)

print("\nNew Flower Prediction")
print("Predicted Cluster:", prediction[0])

# Display cluster results
print("\nCluster Results")
print(df[["Clusters"]].head(20))

# Evaluate model
score = silhouette_score(X_scaled, model.labels_)
print("\nSilhouette Score =", score)

# Visualize clusters
plt.figure(figsize=(8, 5))

plt.scatter(
    df["sepal_length"],
    df["sepal_width"],
    c=df["Clusters"],
    cmap="viridis"
)

plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Iris Flower Clustering using K-Means")
plt.colorbar(label="Cluster")
plt.show()

# Save results
df.to_csv("Iris_Clustering_Results.csv", index=False)

print("Results saved!")
