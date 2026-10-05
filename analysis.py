import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# Load the freefall data (skipping the header line)
data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)

# Extract time (first column) and height y (second column)
t = data[:, 0]
y = data[:, 1]

# --- Part 2: Differentiate to find velocity and acceleration ---
# Calculate velocity (1st derivative of position with respect to time)
v = np.gradient(y, t)

# Calculate acceleration (2nd derivative of position with respect to time)
a = np.gradient(v, t)

# Print the mean acceleration
print("--- Part 2 Results ---")
print(f"Mean Acceleration: {np.mean(a):.4f} m/s^2")
# --- Part 3: Analyze the Effects of Noise ---
print(f"Acceleration Std Dev: {np.std(a):.4f} m/s^2")
# --- Part 4: Integrate acceleration back to position ---
# Recover velocity by integrating acceleration (using initial velocity baseline)
v_recovered = cumulative_trapezoid(a, t, initial=0) + v[0]

# Recover position by integrating the recovered velocity (using initial position baseline)
y_recovered = cumulative_trapezoid(v_recovered, t, initial=0) + y[0]

# Calculate the maximum absolute difference between original and reconstructed position
max_diff = np.max(np.abs(y - y_recovered))

print("\n--- Part 4 Results ---")
print(f"Largest position difference: {max_diff:.4f} meters")
# --- Part 5: Visualize and Report ---
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Top Panel: Position
axs[0].plot(t, y, label='Original Position', color='blue')
axs[0].plot(t, y_recovered, '--', label='Recovered Position', color='orange')
axs[0].set_ylabel('Position (m)')
axs[0].legend()
axs[0].set_title('Motion Data Analysis')

# Middle Panel: Velocity
axs[1].plot(t, v, color='green')
axs[1].set_ylabel('Velocity (m/s)')

# Bottom Panel: Acceleration
axs[2].plot(t, a, color='red')
axs[2].axhline(y=-9.81, color='black', linestyle=':', label='g = -9.81 m/s^2')
axs[2].set_ylabel('Acceleration (m/s^2)')
axs[2].set_xlabel('Time (s)')
axs[2].legend()

plt.tight_layout()
plt.savefig('motion.png')
print("\nPlot saved successfully as 'motion.png'!")

