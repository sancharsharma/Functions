"""
2D functions: Cartesian plane waves and polar (Bessel / power) modes.

Topics:
  - Exp2D plane wave in Cartesian2D and its gradient
  - PolarPower (r^n·e^{imφ}) and PolarBessel (B_m(k·r)·e^{imφ}) modes
  - Evaluating with CoordPoint (auto-converts if needed)
  - Gradient in Polar2D — the φ component uses Bessel recurrences (no 1/r singularity object)
  - numerical_laplacian to check the 2D Helmholtz equation: ∇²f = −k²·f
"""

import numpy as np
import scipy.special as spec
from Functions.Functions_2D import Exp2D, PolarPower, PolarBessel
from Functions.CoordSystems import Cartesian2D, Polar2D, CoordPoint

# ------------------------------------------------------------------ Exp2D (Cartesian plane wave)
# f = 2·e^{i·k·r}, k = (0.5, -0.3)
k = np.array([0.5, -0.3])
f = Exp2D(k_vec=k, ampl=2.0)
print("=== Exp2D: 2·e^{i(0.5 x − 0.3 y)} ===")

r0 = np.array([0.7, 0.4])
print("  f at (x=0.7, y=0.4) =", f(r0))
assert np.isclose(f(r0), 2.0 * np.exp(1j * np.dot(k, r0)))

# Gradient in Cartesian2D is i·k·f
grad_f = f.gradient()
print("  ∇f(r0) =", grad_f(r0), " expected:", 1j * k * f(r0))
assert np.allclose(grad_f(r0), 1j * k * f(r0))

# ------------------------------------------------------------------ PolarPower
# g = 1.5·r²·e^{2iφ}
g = PolarPower(m_azim=2, power=2, ampl=1.5)
print("\n=== PolarPower: 1.5·r²·e^{2iφ} ===")
p0 = np.array([1.2, np.pi/4])   # (r, φ)
print("  g(r=1.2, φ=π/4) =", g(p0))
assert np.isclose(g(p0), 1.5 * 1.2**2 * np.exp(2j * np.pi/4))

dg_r = g.derivative('r')
print("  ∂g/∂r = 3·r·e^{2iφ}, value:", dg_r(p0))
assert np.isclose(dg_r(p0), 1.5 * 2 * 1.2 * np.exp(2j * np.pi/4))

# ------------------------------------------------------------------ PolarBessel
# h = J_1(k_r·r)·e^{iφ}  (first azimuthal mode, Bessel J order 1)
k_r = 0.9
m = 1
h = PolarBessel(m_azim=m, bessel=['J', 1, k_r])
print("\n=== PolarBessel: J_1(k_r·r)·e^{iφ} ===")

# Evaluate via CoordPoint (r, φ)
pt = CoordPoint([1.0, np.pi/4], Polar2D)
val = h(pt)
print("  h at (r=1, φ=π/4) =", val)
expected = spec.jv(1, k_r * 1.0) * np.exp(1j * m * np.pi/4)
assert np.isclose(val, expected), f"{val} != {expected}"

# ∂h/∂r from Bessel recurrence: J_n'(x) = (J_{n-1}(x) − J_{n+1}(x))/2
rho = 1.0
dJ = (spec.jv(0, k_r*rho) - spec.jv(2, k_r*rho)) / 2
expected_dr = k_r * dJ * np.exp(1j*m*np.pi/4)
print("  ∂h/∂r at (1, π/4) =", h.derivative('r')(np.array([rho, np.pi/4])), " expected:", expected_dr)
assert np.isclose(h.derivative('r')(np.array([rho, np.pi/4])), expected_dr, rtol=1e-10)

# φ-component of ∇h = (1/r)·∂h/∂φ — _gradient_component('phi') uses a Bessel recurrence.
grad_h = h.gradient()
r1 = np.array([1.0, np.pi/4])
print("\n=== Gradient ∇h in Polar2D ===")
print("  ∇h(r1) =", grad_h(r1))
assert np.isclose(grad_h(r1)[1], (1j * m / 1.0) * h(r1), rtol=1e-10)
print("  φ-component uses Bessel recurrence — no 1/r singularity object created")

# ------------------------------------------------------------------ Helmholtz check via numerical_laplacian
# For a 2D Bessel mode, ∇²h = −k_r²·h
r_test = np.array([0.8, np.pi/3])   # avoid r=0
lap_num = h.numerical_laplacian(r_test, Polar2D, eps=1e-4)
helm_rhs = -k_r**2 * h(r_test)
print("\n=== Helmholtz check: ∇²h = −k_r²·h ===")
print("  numerical ∇²h =", lap_num)
print("  −k_r²·h       =", helm_rhs)
assert np.isclose(lap_num, helm_rhs, rtol=1e-4), f"Helmholtz mismatch: {lap_num} vs {helm_rhs}"
print("  Helmholtz verified!")

# The same check analytically (no finite differences):
lap_analytic = h.laplacian(Polar2D)(r_test)
assert np.isclose(lap_analytic, helm_rhs, rtol=1e-10)
print("  Analytic laplacian also matches!")

# ------------------------------------------------------------------ Symbolic output
print("\n=== Symbolic output ===")
print("  f.sympy_output() =", f.sympy_output())
print("  g.sympy_output() =", g.sympy_output())
print("  h.sympy_output() =", h.sympy_output())
