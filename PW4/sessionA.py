import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# =====================================================================
# A2. Clean and look
# =====================================================================
print("--- Task A2: Cleaning and Data Profiling ---")

# Load local offline dataset
df = pd.read_csv("data/concrete.csv")

# Count missing values and duplicate rows
missing_count = df.isna().sum().sum()
duplicate_count = df.duplicated().sum()

print(f"Total missing values: {missing_count}")
print(f"Total duplicate rows found: {duplicate_count}")

# Remove duplicates and reset index
df_cleaned = df.drop_duplicates().reset_index(drop=True)
print(f"Shape after removing duplicates: {df_cleaned.shape}")

# Rebuild features (X) and target (y)
X = df_cleaned.drop(columns=["strength"])
y = df_cleaned["strength"]

# Plot and save data distribution profiling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(df_cleaned["cement"], df_cleaned["strength"], alpha=0.5, color="teal")
ax1.set_title("Strength vs Cement Amount")
ax1.set_xlabel("Cement (kg/m³)")
ax1.set_ylabel("Compressive Strength (MPa)")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.scatter(df_cleaned["age"], df_cleaned["strength"], alpha=0.5, color="crimson")
ax2.set_title("Strength vs Curing Age")
ax2.set_xlabel("Age (Days)")
ax2.set_ylabel("Compressive Strength (MPa)")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("concrete_profiles.png")
print("Profile plots saved as 'concrete_profiles.png'")


# =====================================================================
# A3. Split
# =====================================================================
print("\n--- Task A3: Train-Test Split ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training split size: {X_train.shape[0]} rows")
print(f"Testing split size: {X_test.shape[0]} rows")


# =====================================================================
# A4. Linear Regression
# =====================================================================
print("\n--- Task A4: Linear Regression Analysis ---")
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)

# Calculate linear model performance metrics
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

n = len(X_train)
p_linear = X_train.shape[1]
train_adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p_linear - 1)

print(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f} | Train Adjusted R²: {train_adj_r2:.4f} | Test R²: {test_r2:.4f}")

# Plot residuals against age on a log scale
residuals = y_test - y_test_pred
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(X_test["age"], residuals, alpha=0.6, color="darkorange", edgecolor="black")
ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
ax.set_xscale("log")
ax.set_title("Test Residuals vs. Age (Log Scale)")
ax.set_xlabel("Age (Days, Log Scale)")
ax.set_ylabel("Residual Error (MPa)")
ax.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("linear_residuals.png")


# =====================================================================
# A5. Polynomial regression
# =====================================================================
print("\n--- Task A5: Polynomial Regression Sweep ---")
poly_summary_data = []

# Loop correctly through explicit degrees 1, 2, 3, and 4

for deg in [1, 2, 3, 4]:
    poly_pipeline = make_pipeline(
        PolynomialFeatures(degree=deg, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    poly_pipeline.fit(X_train, y_train)
    
    # Extract total generated features
    p_features = poly_pipeline[0].n_output_features_
    
    y_tr_pred = poly_pipeline.predict(X_train)
    y_te_pred = poly_pipeline.predict(X_test)
    
    r_train = np.sqrt(mean_squared_error(y_train, y_tr_pred))
    r_test = np.sqrt(mean_squared_error(y_test, y_te_pred))
    r2_tr = r2_score(y_train, y_tr_pred)
    r2_te = r2_score(y_test, y_te_pred)
    
    n_samples = len(X_train)
    if n_samples - p_features - 1 > 0:
        adj_r2_tr = 1 - (1 - r2_tr) * (n_samples - 1) / (n_samples - p_features - 1)
    else:
        adj_r2_tr = float('nan')
        
    poly_summary_data.append({
        "Degree": deg,
        "Features (p)": p_features,
        "Train RMSE": r_train,
        "Test RMSE": r_test,
        "Train R²": r2_tr,
        "Train Adj R²": adj_r2_tr,
        "Test R²": r2_te
    })

df_poly_table = pd.DataFrame(poly_summary_data)
print("\nPolynomial Degree Comparison Table:")
print(df_poly_table.to_string(index=False))
# =====================================================================
# A6. Classification
# =====================================================================
print("\n--- Task A6: Classification Task (Threshold = 35 MPa) ---")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Convert targets to binary classes: 1 if strength >= 35 else 0
y_train_class = (y_train >= 35).astype(int)
y_test_class = (y_test >= 35).astype(int)

print(f"High-strength samples in training set: {y_train_class.sum()} out of {len(y_train_class)}")
print(f"High-strength samples in testing set: {y_test_class.sum()} out of {len(y_test_class)}")

# 2. KNN Classifier (Requires feature scaling for distance calculations)
print("\n--- Training K-Nearest Neighbors (K=5) ---")
knn_pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
knn_pipeline.fit(X_train, y_train_class)
y_pred_knn = knn_pipeline.predict(X_test)

print(f"KNN Test Accuracy: {accuracy_score(y_test_class, y_pred_knn):.4f}")
print("KNN Classification Report:")
print(classification_report(y_test_class, y_pred_knn))

# 3. Decision Tree Classifier (Does not strictly require feature scaling)
print("\n--- Training Decision Tree (Max Depth = 5) ---")
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train_class)
y_pred_dt = dt_classifier.predict(X_test)

print(f"Decision Tree Test Accuracy: {accuracy_score(y_test_class, y_pred_dt):.4f}")
print("Decision Tree Classification Report:")
print(classification_report(y_test_class, y_pred_dt))
