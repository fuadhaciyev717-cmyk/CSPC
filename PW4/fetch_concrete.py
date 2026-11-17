import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

print("--- Task A1: Fetching Concrete Dataset from UCI ---")

# 1. Fetch the dataset using its official ID
concrete = fetch_ucirepo(id=165)

# 2. Extract features and targets as pandas objects
X = concrete.data.features
y = concrete.data.targets

print("\nPrinting original variables table for validation:")
print(concrete.variables)

# 3. Rename columns cleanly by position to avoid handling long multi-word strings
X = X.copy()
X.columns = ["cement", "slag", "ash", "water", "superplastic", "coarse_agg", "fine_agg", "age"]
y = y.iloc[:, 0].rename("strength")

# 4. Create local data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# 5. Merge features and targets, then save to a local offline CSV file
df_combined = X.assign(strength=y)
df_combined.to_csv("data/concrete.csv", index=False)

print("\n[SUCCESS] Dataset successfully saved locally to 'data/concrete.csv'")
