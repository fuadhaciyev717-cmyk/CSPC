"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO 1: read titration.csv (columns volume_base, pH) into arrays V, pH.

# TODO 2: compute the slope of the pH curve with np.gradient(pH, V), and find
#         the volume where that slope is largest (np.argmax). That is the
#         equivalence point. Print it.

# TODO 3: make two plots side by side: (left) pH vs volume with a line at the
#         equivalence point; (right) the slope vs volume, showing it peaks
#         at the equivalence point. Save as titration.png.
# --- TODO 1: read titration.csv ---
data = np.loadtxt('titration.csv', delimiter=',', skiprows=1)
V = data[:, 0]
pH = data[:, 1]

# --- TODO 2: compute the slope of the pH curve ---
slope = np.gradient(pH, V)
idx_max_slope = np.argmax(slope)
equivalence_volume = V[idx_max_slope]

print("--- Part 5 Results ---")
print(f"Equivalence point volume: {equivalence_volume:.4f} mL")
print(f"Max slope value: {slope[idx_max_slope]:.4f}")

# --- TODO 3: make two plots side by side ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: pH vs volume
ax1.plot(V, pH, marker='o', color='purple', label='pH curve')
ax1.axvline(x=equivalence_volume, color='orange', linestyle='--', label=f'Equivalence ({equivalence_volume:.2f} mL)')
ax1.set_xlabel('Volume of Base Added (mL)')
ax1.set_ylabel('pH')
ax1.set_title('pH vs Volume')
ax1.legend()
ax1.grid(True)

# Right plot: slope vs volume
ax2.plot(V, slope, marker='s', color='teal', label='Slope ($dpH/dV$)')
ax2.axvline(x=equivalence_volume, color='orange', linestyle='--', label=f'Peak ({equivalence_volume:.2f} mL)')
ax2.set_xlabel('Volume of Base Added (mL)')
ax2.set_ylabel('Slope ($dpH/dV$)')
ax2.set_title('Derivative Slope vs Volume')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('titration.png')
print("Side-by-side plots saved successfully as 'titration.png'!")
