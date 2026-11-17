# CSPC - Computer Science for Physics and Chemistry

My course repository. Each practical is under `PW#/Lab #/`.

## Setup
Create the environment for a given lab:
`conda env create -f PW#/Lab #/environment.yml`
`conda activate cspc`

## PW1 - Lab A: Reproducible Simulations

### What I built
* A simulation tracking radioactive decay using array-based operations.
* Unit testing verification modules via Pytest.

### Speed comparison (Loop vs NumPy)
* **Pure Python Loop:** 0.72225 seconds
* **NumPy Version:** 0.00014 seconds
* **Speedup:** NumPy is **5140.5x** faster.

### Tests
* **Status:** All 3 tests passing under `pytest`.

### Conclusion
Using matrix/array libraries like NumPy removes the heavy processing bottlenecks of standard Python loops. This makes physics and chemistry simulations execute thousands of times faster when modeling large quantities of atoms.
## PW1 — Lab B: Data, Plotting, and Automation

### Results
* **Observed Data vs. Analytical Law:** The observed radioactive decay data points match the smooth curve of the analytical law $N_0e^{-\lambda t}$ perfectly on the shared scale, confirming the physical exponential decay profile.
* **Snakemake Pipeline Functionality:** The automated Snakemake pipeline uses timestamp tracking rules to compile data and refresh figures dynamically only when the underlying source code or data arrays change.

## PW2 — Lab A: Motion from Tracking Data

### Results
* **Mean Acceleration:** -8.5797 m/s²
* **Acceleration Standard Deviation:** 28.7161 m/s²
* **Largest Position Reconstructed Difference:** 0.7846 meters

### Analysis
* **Why differentiation amplifies noise:** Numerical differentiation acts like a high-pass filter. It subtracts subsequent data points, meaning that small, high-frequency random measurement errors in the position column get dramatically enlarged with every successive derivative step, dominating the acceleration baseline.
* **Why integration suppresses noise:** Numerical integration works like a low-pass filter. Because integration sums up data points sequentially over time, independent random positive and negative errors cancel each other out over the tracking arc, stabilizing the signal and smoothing the output path.
## PW2 — Lab B: Optimization in Chemistry

### Part 2: Convex and Harder Landscapes
* **Convex Landscape (2A):** All three methods (Gradient Descent, Newton's, and SLSQP) agreed perfectly on the minimum at **x = 3.0000**.
* **Harder Landscape (2B):** From an initial guess of x₀ = 2.0, Gradient Descent and Newton's method converged to a local minimum at **1.1389**, while the SLSQP bounded optimizer successfully reached the stable valley baseline. 

### Part 3: Kinetics Curve Fitting
* **Fitted Rate Constant (k):** 0.2618
* **Plot Output:** Successfully generated `kinetics.png` fitting the experimental exponential decay data cleanly.

### Part 4: Chemical Equilibrium
* **Newton Root Solution:** 0.6638 mol
* **SLSQP Minimization Solution:** 0.6638 mol
* **Equilibrium Yields:** H₂ = 0.3362 mol, I₂ = 0.3362 mol, HI = 1.3277 mol. Both numeric root-finding and square-error optimization methods yield matching equilibrium bounds.

### Part 5: Acid-Base Titration (Bonus)
* **Equivalence Volume Point:** 50.0000 mL
* **Maximum Sloped Gradient:** 4.6000
# PW3 - Data, Distributions, Testing, and Causation

## Session 1 — Characterise the data
### Normality Verdicts
* **age**: Approximately normal ($p = 6.87 \times 10^{-3}$). While the Shapiro-Wilk test rejects strict theoretical normality, visual inspection reveals a fairly symmetric, bell-shaped distribution.
* **chol**: Strongly non-normal ($p = 5.91 \times 10^{-9}$). Right-skewed distribution with severe high-value outliers.
* **trestbps**: Strongly non-normal ($p = 1.80 \times 10^{-6}$). Marked right-skew with a physical lower limit and trailing upper values.
* **thalach**: Approximately normal ($p = 7.00 \times 10^{-5}$). Displays a light left-skew but generally tracks close to an idealized normal curve.

---

## Session 2 — Draw conclusions
### Heart Disease Analysis
* **Max Heart Rate Comparison:** A non-parametric Mann-Whitney U test confirmed a highly significant difference ($p = 1.8010 \times 10^{-13}$). Healthy patients reached a noticeably higher maximum heart rate ($158.38 \pm 1.50$ SEM bpm) than heart-disease patients ($139.26 \pm 1.92$ SEM bpm).
* **Age vs. Heart Rate:** Spearman rank correlation analysis yields a significant negative correlation ($r_s = -0.3916$, $p = 1.5189 \times 10^{-12}$). Maximum heart rate declines progressively as patient age increases.

### Chemical-Exposure Mystery
* **Naive Finding:** Initially, Cadmium appears strongly associated with tumor malignancy due to environmental grouping. 
* **Controlled Finding:** Once restricted to a controlled baseline group (Pollution index between 20 and 40), the Cadmium correlation collapses to $0.2434$, while the true driver, **Benzene**, shows a powerful correlation of $0.7662$.
* **Verdict:** **Benzene is the true causal driver of tumor malignancy.** Cadmium's naive association was entirely spurious, driven by area pollution acting as a prominent confounding variable.
# PW4 — Machine Learning: Supervised and Unsupervised Learning

## Overview
This project explores supervised regression, binary classification, and unsupervised clustering/dimensionality reduction using the **UCI Concrete Compressive Strength dataset** (`id=165`). All preprocessing, modeling, and evaluation steps are separated into reproducible scripts (`fetch_concrete.py`, `sessionA.py`, and `sessionB.py`) running on an offline local dataset (`data/concrete.csv`).

---

## Repository Structure
```text
PW4/
├── data/
│   └── concrete.csv              # Offline cleaned dataset fetched via ucimlrepo
├── fetch_concrete.py             # Online data retrieval and column standardization script
├── sessionA.py                   # Part A: Data cleaning, regression sweep, and classification
├── sessionB.py                   # Part B: K-Means clustering, PCA projection, and Elbow analysis
├── concrete_profiles.png         # Exploratory scatter plots (Strength vs. Cement & Age)
├── linear_residuals.png          # Residual diagnostic plot across curing age (log scale)
├── concrete_clusters_pca.png     # 2D PCA projection colored by K-Means clusters (K=3)
├── elbow_curve.png               # Within-Cluster Sum of Squares (Inertia) across K=2..8
└── README.md                     # Full methodology, metrics, and interpretation report
```

---

## Part A: Supervised Learning

### A1 & A2. Data Acquisition, Cleaning, and Profiling
* **Dataset Dimensions:** The raw dataset contained `1,030` samples across `8` physical/chemical mix features (`cement`, `slag`, `ash`, `water`, `superplastic`, `coarse_agg`, `fine_agg`, `age`) and `1` continuous target (`strength` in MPa).
* **Data Quality:** `0` missing values were detected. `25` identical duplicate rows were identified and dropped, leaving **`1,005` unique samples** (`(1005, 9)`).
* **Why Duplicates Must Be Removed Before Splitting:** Leaving duplicate rows in prior to `train_test_split` causes data leakage. One copy of a duplicated sample can be placed into the training set while its identical twin ends up in the test set. During evaluation, the model achieves artificially low error on that test point through rote memorization rather than true generalization, producing an over-optimistic evaluation that hides overfitting.
* **Exploratory Trends (`concrete_profiles.png`):** Compressive strength shows a strong positive linear relationship with `cement` content and a non-linear, logarithmic growth curve with `age` (rapid strength gain in the first 28 days, tapering off at 90–365 days).

### A3 & A4. Train-Test Split & Baseline Linear Regression
* **Split Strategy:** `80%` training (`804` samples) and `20%` testing (`201` samples) with `random_state=42`.
* **Baseline Linear Model Performance:**
  * **Train RMSE:** `10.0179 MPa` | **Test RMSE:** `11.1913 MPa`
  * **Train $R^2$:** `0.6099` | **Train Adjusted $R^2$:** `0.6059` | **Test $R^2$:** `0.5802`
* **Residual Diagnostics (`linear_residuals.png`):** Plotting test residuals against curing `age` on a logarithmic scale reveals a distinct curved, non-random pattern rather than uniform random scatter around zero. Because concrete cures logarithmically while a degree-1 linear model assumes a constant rate of strength gain per day, the linear model systematically under-predicts medium-age concrete and over-predicts very early/late curing ages.

### A5. Polynomial Regression Complexity Sweep
Features were expanded using `PolynomialFeatures` (`degree=1` to `4`) and standardized via `StandardScaler` inside a `scikit-learn` pipeline to prevent scale distortion across high-order interaction terms:

| Degree | Features ($p$) | Train RMSE (MPa) | Test RMSE (MPa) | Train $R^2$ | Train Adj $R^2$ | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Linear)** | 8 | 10.0179 | 11.1913 | 0.6099 | 0.6059 | 0.5802 |
| **2** | 44 | 6.9802 | 7.9133 | 0.8105 | 0.7994 | 0.7896 |
| **3** | 164 | 4.3900 | **5.9515** | 0.9246 | 0.9053 | **0.8813** |
| **4 (Overfit)** | 494 | **2.3339** | 39.4096 | **0.9788** | 0.9448 | -4.2061 |

* **Complexity & Overfitting Analysis:**
  * Moving from **Degree 1 to Degree 3** captures the non-linear interactions between water, cement, additives, and curing time, reducing Test RMSE from `11.19 MPa` down to **`5.95 MPa`** ($R^2 = 0.8813$).
  * At **Degree 4**, the polynomial expansion generates **`494` features** for only `804` training rows ($p$ approaches $n$). The model memorizes high-frequency training noise, driving Train RMSE down to `2.33 MPa` while Test RMSE explodes to **`39.41 MPa`** ($R^2 = -4.21$)—a textbook demonstration of catastrophic overfitting.

### A6. Binary Classification ($\ge 35\text{ MPa}$ High-Strength Concrete)
The continuous `strength` target was converted into a binary label (`1` if $\text{strength} \ge 35\text{ MPa}$, `0` otherwise), resulting in `98` high-strength and `103` standard-strength samples in the test set:
* **K-Nearest Neighbors (`K=5`, with `StandardScaler`):** Requires feature scaling so Euclidean distances are not dominated by high-magnitude variables like `coarse_agg` (`~1000 kg/m³`) over `superplastic` (`~0–30 kg/m³`).
* **Decision Tree (`max_depth=5`, `random_state=42`):** Achieved **`85.07%` test accuracy**, with balanced precision (`0.85`), recall (`0.85`), and F1-score (`0.85`) across both classes. Tree-based splits are invariant to monotonic feature scaling and effectively isolate non-linear mix thresholds.

---

## Part B: Unsupervised Learning

### B1 & B2. K-Means Clustering & PCA Visualization
* **Preprocessing:** The `strength` target column was dropped so clustering relied purely on mix ingredients and curing age. All 8 features were standardized using `StandardScaler` prior to distance calculations.
* **Principal Component Analysis (`concrete_clusters_pca.png`):**
  * **PC1 Explained Variance:** `28.90%`
  * **PC2 Explained Variance:** `17.86%`
  * **Total Explained Variance (Top 2 Components):** **`46.76%`**
* **Cluster Structure ($K=3$):** Projecting the standardized dataset onto the top two principal components shows three distinct, well-separated groupings (`423` samples in Cluster 0, `316` in Cluster 1, and `266` in Cluster 2).

### B3. Optimal $K$ Selection & Physical Interpretation
* **Elbow Method (`elbow_curve.png`):** Sweeping $K$ from `2` to `8` shows a steep reduction in inertia (Within-Cluster Sum of Squares) up to $K=3$ (`5476.81`) and $K=4$ (`4754.18`), followed by a flatter curve of diminishing returns (`4068.63` at $K=5$ down to `3161.40` at $K=8$), confirming $K=3$–$4$ as the natural structural partitioning of the dataset.
* **Physical Meaning of Discovered Clusters ($K=3$):** Even without access to `strength` during training, K-Means separated the recipes into three real-world concrete engineering formulations:

| Cluster | Sample Count | Cement ($\text{kg/m}^3$) | Slag ($\text{kg/m}^3$) | Fly Ash ($\text{kg/m}^3$) | Water ($\text{kg/m}^3$) | Superplastic ($\text{kg/m}^3$) | Age (Days) | Avg Strength (MPa) | Formulation Profile |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | 423 | 263.84 | 32.43 | **109.62** | **167.91** | **10.43** | 35.43 | **36.96** | **Modern Additive Mix:** Low water, high fly ash & superplasticizer; achieves highest average strength. |
| **1** | 316 | 225.55 | **177.29** | 26.40 | 190.46 | 4.73 | 36.69 | **34.14** | **Slag-Blended Mix:** Heavy blast-furnace slag replacement with reduced pure cement content. |
| **2** | 266 | **366.49** | 10.02 | 4.15 | **194.64** | 0.59 | **73.32** | **33.85** | **Traditional Pure-Cement Mix:** High cement & water, virtually no chemical/mineral additives, longer curing times. |

---

## How to Reproduce
```bash
# 1. Activate environment
conda activate cspc

# 2. Fetch and save offline dataset (requires internet)
python fetch_concrete.py

# 3. Run Part A (Supervised Regression & Classification)
python sessionA.py

# 4. Run Part B (Unsupervised K-Means & PCA)
python sessionB.py
```


