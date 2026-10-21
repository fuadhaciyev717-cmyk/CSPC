import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load the data using the exact filename in your directory
df = pd.read_csv("Heart_disease_cleveland_new.csv")
continuous_cols = ["age", "chol", "trestbps", "thalach"]

# Print baseline dataset details to terminal
print("--- Dataset Information ---")
print(f"Dataset Shape: {df.shape}\n")
print("First 5 rows of the dataset:")
print(df.head(), "\n")

# 2 & 3. Visualise distributions and perform Normality Tests
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

print("--- Normality Test Results (Shapiro-Wilk) ---")
for i, col in enumerate(continuous_cols):
    # Plot empirical histogram
    axes[i].hist(df[col], bins=20, edgecolor='black', alpha=0.7, density=True)
    
    # Overlay idealized normal curve distribution for visual reference
    mu, std = df[col].mean(), df[col].std()
    xmin, xmax = axes[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[i].plot(x, p, 'r-', linewidth=2)
    
    # Label formatting
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
    
    # Statistical test: Shapiro-Wilk
    stat, p_val = stats.shapiro(df[col])
    print(f"{col:<8} -> Stat: {stat:.4f}, p-value: {p_val:.4e}")

plt.tight_layout()
plt.savefig("distributions.png")
print("\nPlot successfully saved as 'distributions.png'")
