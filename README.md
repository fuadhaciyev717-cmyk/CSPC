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
