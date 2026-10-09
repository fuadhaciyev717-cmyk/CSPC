"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize

# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)
#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")
# --- TODO 2A: minimize f three ways from x0=0 ---
x0 = 0.0

# (1) Gradient descent by hand
x_gd = x0
lr = 0.1  # learning rate step size
for _ in range(1000):
    grad = df(x_gd)
    if abs(grad) < 1e-6:  # stop when step is tiny
        break
    x_gd = x_gd - lr * grad
print(f"2A - Gradient Descent Result: {x_gd:.4f}")

# (2) scipy.optimize.newton on f'(x) with fprime
x_newton = import_newton = newton(df, x0, fprime=d2f)
print(f"2A - Newton Method Result: {x_newton:.4f}")

# (3) scipy.optimize.minimize using SLSQP
res_slsqp = minimize(f, x0, method="SLSQP")
print(f"2A - SLSQP Result: {res_slsqp.x[0]:.4f}\n")


# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?
# --- TODO 2B: run the same three methods on g, from x0=0 AND from x0=2 ---
for x0 in [0.0, 2.0]:
    print(f"--- Running 2B with initial guess x0 = {x0} ---")
    
    # (1) Gradient descent by hand
    x_gd = x0
    lr = 0.01  # Smaller step size for a steeper polynomial landscape
    for _ in range(2000):
        grad = dg(x_gd)
        if abs(grad) < 1e-6:
            break
        x_gd = x_gd - lr * grad
    print(f"2B - Gradient Descent Result: {x_gd:.4f}")
    
    # (2) scipy.optimize.newton
    try:
        x_newton = newton(dg, x0, fprime=d2g)
        # Check curvature sign to see if it's a minimum or maximum
        curvature = d2g(x_newton)
        nature = "minimum" if curvature > 0 else "maximum"
        print(f"2B - Newton Method Result: {x_newton:.4f} ({nature}, g'' = {curvature:.4f})")
    except Exception as e:
        print(f"2B - Newton Method failed from x0 = {x0}")
        
    # (3) scipy.optimize.minimize using SLSQP
    res_slsqp = minimize(g, x0, method="SLSQP")
    print(f"2B - SLSQP Result: {res_slsqp.x[0]:.4f}\n")

