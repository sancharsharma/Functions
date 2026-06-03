"""
3D plane waves and Cartesian gradient/Laplacian.

Topics:
  - Exp3D construction and evaluation
  - Partial derivatives via .derivative('x') etc.
  - Gradient via Cartesian3D.gradient(f) → VecFunc
  - Laplacian via Cartesian3D.laplacian(f)
  - Analytical check: ∇²(e^{ik·r}) = -|k|² · e^{ik·r}
"""

import numpy as np
from Functions.Functions_3D import Exp3D
from Functions.CoordSystems import Cartesian3D

# ------------------------------------------------------------------ Construction
k = np.array([1.0, 2.0, -1.0])
f = Exp3D(k_vec=k, ampl=1)
print("=== Exp3D: exp(i·k·r), k =", k, "===")

# Evaluate at a single point (3-vector → scalar)
r0 = np.array([1.0, 0.5, 2.0])
val = f(r0)
expected = np.exp(1j * np.dot(k, r0))
print("  f(r0)    =", val)
print("  expected =", expected)
assert np.isclose(val, expected)

# Evaluate at multiple points (N×3 array → length-N array)
pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
vals = f(pts)
print("  f(pts)   =", vals)
assert np.allclose(vals, np.exp(1j * (pts @ k)))

# ------------------------------------------------------------------ Partial derivatives
# ∂f/∂x = i·k_x · f
df_x = f.derivative('x')
df_y = f.derivative('y')
df_z = f.derivative('z')

print("\n=== Partial derivatives ===")
print("  ∂f/∂x at r0 =", df_x(r0), " expected:", 1j*k[0]*f(r0))
print("  ∂f/∂y at r0 =", df_y(r0), " expected:", 1j*k[1]*f(r0))
print("  ∂f/∂z at r0 =", df_z(r0), " expected:", 1j*k[2]*f(r0))
assert np.isclose(df_x(r0), 1j*k[0]*f(r0))
assert np.isclose(df_y(r0), 1j*k[1]*f(r0))
assert np.isclose(df_z(r0), 1j*k[2]*f(r0))

# ------------------------------------------------------------------ Gradient
# ∇f = i·k · f  (returned as VecFunc with 3 components)
grad_f = Cartesian3D.gradient(f)
from Functions.Functions_Base import VecFunc
print("\n=== Gradient ∇f ===")
print("  type:", type(grad_f).__name__)
grad_at_r0 = grad_f(r0)      # (3,) complex array
print("  ∇f(r0) =", grad_at_r0)
expected_grad = 1j * k * f(r0)
print("  expected:", expected_grad)
assert np.allclose(grad_at_r0, expected_grad)

# ------------------------------------------------------------------ Laplacian
# ∇²f = div(∇f).  divergence() expects a list of components, so pass grad_f.components.
lap_f = Cartesian3D.divergence(grad_f.components)
lap_at_r0 = lap_f(r0)
print("\n=== Laplacian ∇²f ===")
print("  ∇²f(r0)   =", lap_at_r0)
print("  -|k|²·f   =", -np.dot(k, k) * f(r0))
assert np.isclose(lap_at_r0, -np.dot(k, k) * f(r0), rtol=1e-10)

# Verify at several points
lap_at_pts = lap_f(pts)
f_at_pts   = f(pts)
assert np.allclose(lap_at_pts, -np.dot(k, k) * f_at_pts, rtol=1e-10)
print("  Laplacian = -|k|²·f verified at", len(pts), "points")

# ------------------------------------------------------------------ Symbolic output
print("\n=== Symbolic output ===")
print("  f.sympy_output() =", f.sympy_output())
print("  df_x.sympy_output() =", df_x.sympy_output())
