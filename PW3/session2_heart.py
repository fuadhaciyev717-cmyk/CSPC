import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Load dataset
df = pd.read_csv("Heart_disease_cleveland_new.csv")

# =====================================================================
# 4. Is the effect real? (thalach comparison)
# =====================================================================
print("--- Task 4: Heart Rate Comparison ---")
# Split thalach into two groups based on target (0 = healthy, 1 = disease)
healthy = df[df["target"] == 0]["thalach"]
disease = df[df["target"] == 1]["thalach"]

# Non-parametric test because thalach is non-normal
u_stat, p_val_u = stats.mannwhitneyu(healthy, disease, alternative='two-sided')
print(f"Mann-Whitney U test p-value: {p_val_u:.4e}")

# Calculate means and Standard Error of the Mean (SEM) for uncertainty
mean_h, sem_h = healthy.mean(), healthy.sem()
mean_d, sem_d = disease.mean(), disease.sem()

print(f"Healthy group mean max heart rate: {mean_h:.2f} (±{sem_h:.2f} SEM)")
print(f"Disease group mean max heart rate: {mean_d:.2f} (±{sem_d:.2f} SEM)")

# Produce the uncertainty bar plot
plt.figure(figsize=(6, 5))
plt.bar(["Healthy (0)", "Disease (1)"], [mean_h, mean_d], 
        yerr=[sem_h, sem_d], capsize=10, color=['skyblue', 'lightcoral'], edgecolor='black', alpha=0.8)
plt.ylabel("Mean Maximum Heart Rate (thalach)")
plt.title("Mean thalach with Uncertainty (±1 SEM)")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("thalach_comparison.png")
print("Saved comparison plot as 'thalach_comparison.png'\n")

# =====================================================================
# 5. Do two things move together? (age vs thalach)
# =====================================================================
print("--- Task 5: Age vs Max Heart Rate Correlation ---")
# Calculate Spearman rank correlation (robust to non-normal data)
corr, p_val_corr = stats.spearmanr(df["age"], df["thalach"])
print(f"Spearman correlation coefficient: {corr:.4f} (p-value: {p_val_corr:.4e})")

# Plot age vs thalach scatter plot
plt.figure(figsize=(7, 5))
plt.scatter(df["age"], df["thalach"], alpha=0.6, color='purple', edgecolor='none')
# Add a linear trendline
m, b = np.polyfit(df["age"], df["thalach"], 1)
plt.plot(df["age"], m*df["age"] + b, color='red', linestyle='--', linewidth=2, label=f'Trendline (r={corr:.2f})')

plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate (thalach)")
plt.title("Relationship between Age and Maximum Heart Rate")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig("age_vs_thalach.png")
print("Saved correlation plot as 'age_vs_thalach.png'")
