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

