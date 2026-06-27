"""
Cylindrical Bessel modes and curvilinear operators.

Topics:
  - Cylindrical construction (Bessel J/Y/I/K) and evaluation
  - PowerCylindrical as a simpler cylindrical mode
  - Evaluating with CoordPoint (auto-converts if needed)
  - Gradient in Cylindrical3D — the φ component uses Bessel recurrences
  - numerical_laplacian to check the Helmholtz equation: ∇²f = −k²·f
"""

import numpy as np
from Functions.Functions_3D import Cylindrical, PowerCylindrical
from Functions.CoordSystems import Cylindrical3D, CoordPoint

# ------------------------------------------------------------------ Cylindrical
# f = J_1(k_r·ρ) · e^{iφ} · e^{ik_z·z}  (first azimuthal mode, Bessel J order 1)
k_r = 2.4048   # first zero of J_0 — chosen so J_0(k_r) ≈ 0
kz  = 1.0
m   = 1        # azimuthal mode number

f = Cylindrical(kz=kz, m_azim=m, bessel=['J', 1, k_r])
print("=== Cylindrical: J_1(k_r·ρ)·e^{iφ}·e^{ik_z·z} ===")

# Evaluate via CoordPoint (ρ, φ, z)
pt = CoordPoint([1.0, np.pi/4, 0.5], Cylindrical3D)
val = f(pt)
print("  f at (ρ=1, φ=π/4, z=0.5) =", val)

# Raw array evaluation for verification
import scipy.special as spec
rho, phi, z = 1.0, np.pi/4, 0.5
expected = spec.jv(1, k_r * rho) * np.exp(1j * m * phi) * np.exp(1j * kz * z)
assert np.isclose(val, expected), f"{val} != {expected}"

# ------------------------------------------------------------------ Derivatives
print("\n=== Derivatives ===")
df_rho = f.derivative('rho')
df_phi = f.derivative('phi')
df_z   = f.derivative('z')

r0 = np.array([rho, phi, z])
print("  ∂f/∂z  at r0 =", df_z(r0), " expected:", 1j*kz*f(r0))
assert np.isclose(df_z(r0), 1j*kz*f(r0))

print("  ∂f/∂φ  at r0 =", df_phi(r0), " expected:", 1j*m*f(r0))
assert np.isclose(df_phi(r0), 1j*m*f(r0))

# ∂f/∂ρ comes from Bessel recurrence: J_n'(x) = (J_{n-1}(x) - J_{n+1}(x))/2
dJ = (spec.jv(0, k_r*rho) - spec.jv(2, k_r*rho)) / 2
expected_drho = k_r * dJ * np.exp(1j*m*phi) * np.exp(1j*kz*z)
print("  ∂f/∂ρ  at r0 =", df_rho(r0), " expected:", expected_drho)
assert np.isclose(df_rho(r0), expected_drho, rtol=1e-10)

# ------------------------------------------------------------------ Gradient
# φ-component of ∇f in cylindrical coords = (1/ρ)·∂f/∂φ
# _gradient_component('phi') uses Bessel recurrences to avoid creating a CoordPow.
grad_f = f.gradient()
from Functions.Functions_Base import VecFunc
print("\n=== Gradient ∇f in Cylindrical3D ===")
grad_val = grad_f(r0)     # (3,) array: [∂f/∂ρ, (1/ρ)·∂f/∂φ, ∂f/∂z]
print("  ∇f(r0) =", grad_val)
expected_phi_comp = (1j * m / rho) * f(r0)
assert np.isclose(grad_val[1], expected_phi_comp, rtol=1e-10)
print("  φ-component uses Bessel recurrence — no 1/ρ singularity object created")

# ------------------------------------------------------------------ PowerCylindrical
# g = ρ^2 · e^{2iφ} · e^{ik_z·z}
g = PowerCylindrical(kz=1.0, m_azim=2, power=2)
print("\n=== PowerCylindrical: ρ²·e^{2iφ}·e^{iz} ===")
print("  g(r0) =", g(r0))
expected_g = rho**2 * np.exp(2j*phi) * np.exp(1j*z)
assert np.isclose(g(r0), expected_g)

dg_rho = g.derivative('rho')
print("  ∂g/∂ρ = 2ρ·e^{2iφ}·e^{iz}, value:", dg_rho(r0))
assert np.isclose(dg_rho(r0), 2*rho * np.exp(2j*phi) * np.exp(1j*z))

# ------------------------------------------------------------------ Helmholtz check via numerical_laplacian
# For a Bessel mode, ∇²f = (-(k_r²+kz²))·f
rho_test = 0.8   # avoid ρ=0 singularity
r_test = np.array([rho_test, np.pi/3, 1.0])
lap_num = f.numerical_laplacian(r_test, Cylindrical3D, eps=1e-4)
helm_rhs = -(k_r**2 + kz**2) * f(r_test)
print("\n=== Helmholtz check: ∇²f = −(k_r²+k_z²)·f ===")
print("  numerical ∇²f       =", lap_num)
print("  −(k_r²+k_z²)·f      =", helm_rhs)
assert np.isclose(lap_num, helm_rhs, rtol=1e-4), f"Helmholtz mismatch: {lap_num} vs {helm_rhs}"
print("  Helmholtz verified!")

# ------------------------------------------------------------------ Batch CoordPoint input
print("\n=== Batch CoordPoint input ===")
batch_pts = [
    CoordPoint([0.5, 0.0,     0.0], Cylindrical3D),
    CoordPoint([1.0, np.pi/4, 0.5], Cylindrical3D),
    CoordPoint([1.5, np.pi/2, 1.0], Cylindrical3D),
]
batch_vals = f(batch_pts)
expected_batch = np.array([f(p) for p in batch_pts])
assert np.allclose(batch_vals, expected_batch), f"batch mismatch: {batch_vals} vs {expected_batch}"
print("  f on 3 points:", batch_vals)
print("  Batch CoordPoint input verified!")

# ------------------------------------------------------------------ Symbolic output
print("\n=== Symbolic output ===")
print("  f.sympy_output() =", f.sympy_output())
print("  g.sympy_output() =", g.sympy_output())
