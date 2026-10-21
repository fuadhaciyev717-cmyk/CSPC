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


