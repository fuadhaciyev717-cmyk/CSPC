"""
PW2 Lab B Part 3 -- fit a reaction's rate constant to measured data.

A first-order reaction decays as  C(t) = C0 * exp(-k*t).  You have noisy
concentration-vs-time measurements; find the k that best matches them.
Complete the TODOs. Run:  python kinetics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# TODO 1: read kinetics.csv (columns time, concentration) into arrays t, C.
#         Set C0 = the first concentration.

# TODO 2: write total_error(k) = sum of (measured - C0*exp(-k*t))^2.
#         This is the "how bad" number: small when the model matches the data.

# TODO 3: minimise total_error with scipy.optimize.minimize (method "SLSQP",
#         bounds [(0, 5)], start x0=0.5). Print the fitted k.

# TODO 4: plot the measured data (points) and your fitted curve (line) together.
#         Save as kinetics.png.
# --- TODO 1: Read kinetics.csv and set C0 ---
data = np.loadtxt('kinetics.csv', delimiter=',', skiprows=1)
t = data[:, 0]
C = data[:, 1]
C0 = C[0]

# --- TODO 2: Write total_error(k) ---
def total_error(k):
    C_model = C0 * np.exp(-k * t)
    return np.sum((C - C_model) ** 2)

# --- TODO 3: Minimize total_error with SLSQP ---
res = minimize(total_error, x0=0.5, method="SLSQP", bounds=[(0, 5)])
fitted_k = res.x[0]
print("--- Part 3 Results ---")
print(f"Fitted rate constant k: {fitted_k:.4f}")

# --- TODO 4: Plot measured data and fitted curve ---
plt.figure(figsize=(7, 5))
plt.scatter(t, C, color='red', label='Measured Data')
t_fine = np.linspace(t.min(), t.max(), 200)
C_fine = C0 * np.exp(-fitted_k * t_fine)
plt.plot(t_fine, C_fine, color='blue', label=f'Fitted Curve (k={fitted_k:.3f})')
plt.xlabel('Time (s)')
plt.ylabel('Concentration')
plt.title('First-Order Kinetics Optimization')
plt.legend()
plt.grid(True)
plt.savefig('kinetics.png')
print("Plot successfully saved as 'kinetics.png'!")
