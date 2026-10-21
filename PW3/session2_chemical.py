import pandas as pd
import scipy.stats as stats

# Load Dataset 2
df_chem = pd.read_csv("chemicals_cancer.csv")

print("=====================================================================")
print("--- Task 6: Naive Analysis (Whole Group Correlation) ---")
print("=====================================================================")
# Compute global correlation between chemicals and tumor malignancy score
corr_benzene_naive, _ = stats.spearmanr(df_chem["benzene"], df_chem["malignancy"])
corr_cadmium_naive, _ = stats.spearmanr(df_chem["cadmium"], df_chem["malignancy"])

print(f"Naive Benzene vs Malignancy correlation: {corr_benzene_naive:.4f}")
print(f"Naive Cadmium vs Malignancy correlation: {corr_cadmium_naive:.4f}")

print("\n=====================================================================")
print("--- Task 7: Confounder Analysis (Controlled Stratification) ---")
print("=====================================================================")
# The dataset contains columns like area 'pollution_index' and 'age'.
# Let's check which variables associate with each other to spot the backend driver.
print("Checking correlation with pollution_index:")
print(f"Benzene vs Pollution: {stats.spearmanr(df_chem['benzene'], df_chem['pollution_index'])[0]:.4f}")
print(f"Cadmium vs Pollution: {stats.spearmanr(df_chem['cadmium'], df_chem['pollution_index'])[0]:.4f}")
print(f"Malignancy vs Pollution: {stats.spearmanr(df_chem['malignancy'], df_chem['pollution_index'])[0]:.4f}")

# Simple approach: Restrict data to a subset of patients with a similar pollution_index profile
# This filters out the impact of pollution to let us look at the pure chemical effect.
low_pollution_slice = df_chem[(df_chem["pollution_index"] > 20) & (df_chem["pollution_index"] < 40)]

corr_benzene_controlled, _ = stats.spearmanr(low_pollution_slice["benzene"], low_pollution_slice["malignancy"])
corr_cadmium_controlled, _ = stats.spearmanr(low_pollution_slice["cadmium"], low_pollution_slice["malignancy"])

print(f"\nControlled Slice Correlation (Pollution restricted between 20 and 40, Sample Size: {len(low_pollution_slice)}):")
print(f"Controlled Benzene vs Malignancy: {corr_benzene_controlled:.4f}")
print(f"Controlled Cadmium vs Malignancy: {corr_cadmium_controlled:.4f}")
