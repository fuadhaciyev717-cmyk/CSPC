import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("=====================================================================")
print("--- Part B: Unsupervised Learning (K-Means & PCA) ---")
print("=====================================================================")

# 1. Load the cleaned data from Part A
df = pd.read_csv("data/concrete.csv").drop_duplicates().reset_index(drop=True)

# Unsupervised learning uses features only (drop the target 'strength')
X = df.drop(columns=["strength"])

# 2. Scale the features (Crucial for distance-based algorithms like K-Means and PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Task B1: Apply K-Means Clustering (Let's use K=3 clusters as a baseline)
k_clusters = 3
kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42, n_init=10)
cluster_labels = kmeans.fit_transform(X_scaled).argmin(axis=1) # Get cluster assignments
df["cluster"] = cluster_labels

print(f"[SUCCESS] K-Means clustering completed for K={k_clusters}")
print("Sample distribution per cluster:")
print(df["cluster"].value_counts())

# 4. Task B2: Principal Component Analysis (PCA) dimensional reduction
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Calculate variance explained by the first two components
variance_ratio = pca.explained_variance_ratio_
print(f"\nPCA Explained Variance: PC1 = {variance_ratio[0]:.4f}, PC2 = {variance_ratio[1]:.4f}")
print(f"Total Explained Variance (Top 2 Components): {sum(variance_ratio)*100:.2f}%")

# 5. Plotting the PCA space colored by K-Means Cluster labels
plt.figure(figsize=(9, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, edgecolor='k', s=40)

# Add layout aesthetics
plt.title(f"2D PCA Projection of Concrete Features Colored by K-Means (K={k_clusters})")
plt.xlabel(f"Principal Component 1 ({variance_ratio[0]*100:.1f}%)")
plt.ylabel(f"Principal Component 2 ({variance_ratio[1]*100:.1f}%)")
plt.grid(True, linestyle="--", alpha=0.5)

# Add a legend for the cluster groups
legend = plt.legend(*scatter.legend_elements(), title="Clusters")
plt.gca().add_artist(legend)

plt.tight_layout()
plt.savefig("concrete_clusters_pca.png")
print("\n[SUCCESS] PCA Cluster visualization saved as 'concrete_clusters_pca.png'")
# =====================================================================
# Task B3: Optimal K (Elbow Method) & Cluster Profiling
# =====================================================================
print("\n--- Task B3: Elbow Method Sweep (K = 2 to 8) ---")

inertias = []
k_values = [2, 3, 4, 5, 6, 7, 8]

for k in k_values:
    km = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"K = {k} -> Inertia (WCSS): {km.inertia_:.2f}")

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker='o', linestyle='-', color='darkblue', linewidth=2)
plt.title("K-Means Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.xticks(k_values)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("elbow_curve.png")
print("\n[SUCCESS] Elbow plot saved as 'elbow_curve.png'")

# Inspect mean feature & strength values for our K=3 clusters
print("\n--- Mean Feature Values per Cluster (K=3) ---")
cluster_profile = df.groupby("cluster").mean()
print(cluster_profile.round(2).to_string())
