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

