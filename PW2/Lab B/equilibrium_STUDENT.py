"""
PW2 Lab B Part 4 -- chemical equilibrium via the equilibrium constant K.

Reaction  H2 + I2 <=> 2 HI, starting from 1 mol H2 and 1 mol I2.
As the reaction proceeds by an extent x:  H2 = 1-x,  I2 = 1-x,  HI = 2x.
At equilibrium the composition satisfies the equilibrium constant
        K = [HI]^2 / ([H2][I2]) = (2x)^2 / ((1-x)(1-x)).
Given K, find the extent x. Solve it TWO ways and compare.
Run:  python equilibrium.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.6
a = b = 1.0

# TODO 1: write k_imbalance(x) = (2x)^2/((a-x)(b-x)) - K.
#         It equals zero exactly at equilibrium.

# TODO 2 (method 1): use scipy.optimize.newton to find the root of k_imbalance
#         (start x0=0.5). This is root-finding.

# TODO 3 (method 2): use scipy.optimize.minimize to minimise k_imbalance(x)**2
#         (method "SLSQP", bounds [(0, 0.999)], x0=[0.5]). Print both answers
#         and confirm they agree.

# TODO 4: report the equilibrium amounts (H2, I2, HI), and plot how the three
#         amounts change with the extent x, marking the equilibrium. Save
#         equilibrium.png.
# --- TODO 1: write k_imbalance(x) ---
def k_imbalance(x):
    return ((2.0 * x) ** 2) / ((a - x) * (b - x)) - K

# --- TODO 2 (method 1): use scipy.optimize.newton ---
x_newton = newton(k_imbalance, x0=0.5)

# --- TODO 3 (method 2): use scipy.optimize.minimize ---
def objective_minimize(x):
    return k_imbalance(x[0]) ** 2

res = minimize(objective_minimize, x0=[0.5], method="SLSQP", bounds=[(0.0, 0.999)])
x_slsqp = res.x[0]

print("--- Part 4 Results ---")
print(f"Newton root solution x: {x_newton:.4f}")
print(f"SLSQP minimization solution x: {x_slsqp:.4f}")

# --- TODO 4: report the equilibrium amounts and plot ---
H2_eq = a - x_newton
I2_eq = b - x_newton
HI_eq = 2.0 * x_newton

print(f"\nEquilibrium Amounts:")
print(f"H2: {H2_eq:.4f} mol")
print(f"I2: {I2_eq:.4f} mol")
print(f"HI: {HI_eq:.4f} mol")

# Generate chemical equilibrium trend plot
x_vals = np.linspace(0, 0.99, 100)
plt.figure(figsize=(7, 5))
plt.plot(x_vals, a - x_vals, label='$H_2$', color='blue')
plt.plot(x_vals, b - x_vals, label='$I_2$', color='green')
plt.plot(x_vals, 2.0 * x_vals, label='$HI$', color='red')
plt.axvline(x=x_newton, color='orange', linestyle='--', label=f'Equilibrium (x={x_newton:.3f})')
plt.xlabel('Reaction Extent $x$ (mol)')
plt.ylabel('Amount (mol)')
plt.title('Chemical Equilibrium Amounts vs Reaction Extent')
plt.legend()
plt.grid(True)
plt.savefig('equilibrium.png')
print("\nPlot successfully saved as 'equilibrium.png'!")
