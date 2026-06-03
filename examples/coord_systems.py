"""
Coordinate systems, CoordPoint, and cross-system comparisons.

Topics:
  - CoordPoint: tagging a position with a coordinate system
  - Converting between Cartesian3D, Cylindrical3D, and Polar3D
  - Gradient in different coordinate systems on the same scalar field
  - Laplacian in Cartesian3D vs Cylindrical3D, compared numerically
"""

import numpy as np
from Functions.CoordSystems import Cartesian3D, Cylindrical3D, Polar3D, CoordPoint

# ------------------------------------------------------------------ CoordPoint and conversion
# A point at (x=1, y=1, z=0) in Cartesian
cart_pt = CoordPoint([1.0, 1.0, 0.0], Cartesian3D)
print("=== CoordPoint conversions ===")
print("  Cartesian:   ", cart_pt)

# Convert to cylindrical (ρ, φ, z)
cyl_pt = cart_pt.convert_to(Cylindrical3D)
print("  Cylindrical: ", cyl_pt)
rho_expected = np.sqrt(2)
phi_expected = np.pi / 4
assert np.isclose(cyl_pt.pos[0], rho_expected)
assert np.isclose(cyl_pt.pos[1], phi_expected)
assert np.isclose(cyl_pt.pos[2], 0.0)

# Convert to spherical (r, θ, φ)
sph_pt = cart_pt.convert_to(Polar3D)
print("  Spherical:   ", sph_pt)
r_expected = np.sqrt(2)
theta_expected = np.pi / 2   # equatorial plane (z=0)
assert np.isclose(sph_pt.pos[0], r_expected)
assert np.isclose(sph_pt.pos[1], theta_expected)

# Round-trip: Cylindrical → Cartesian should recover original
cart_back = cyl_pt.convert_to(Cartesian3D)
print("  Round-trip:  ", cart_back)
assert np.allclose(cart_back.pos, cart_pt.pos, atol=1e-12)

# ------------------------------------------------------------------ Gradient: Cartesian vs Cylindrical
# Use Exp3D (Cartesian) as the test field: f = exp(i·k·r), k = (1,0,0)
from Functions.Functions_3D import Exp3D, PowerCylindrical

print("\n=== Gradient: Exp3D in Cartesian3D ===")
k = np.array([1.0, 0.0, 0.0])
f_cart = Exp3D(k_vec=k)
grad_cart = Cartesian3D.gradient(f_cart)
r0_cart = np.array([0.5, 0.3, 0.1])
gv = grad_cart(r0_cart)
print("  ∇f(r0) =", gv)
print("  expected i·k·f:", 1j * k * f_cart(r0_cart))
assert np.allclose(gv, 1j * k * f_cart(r0_cart))

# Use PowerCylindrical as the test field: g = ρ^1·e^{0·φ}·e^{0·z} = ρ
print("\n=== Gradient: PowerCylindrical(ρ) in Cylindrical3D ===")
g_cyl = PowerCylindrical(kz=0, m_azim=0, power=1)  # g = ρ
grad_cyl = Cylindrical3D.gradient(g_cyl)
r0_cyl = np.array([1.5, np.pi/3, 0.0])   # (ρ, φ, z)
gv2 = grad_cyl(r0_cyl)
print("  ∇ρ(r0) =", gv2, " [expected (1, 0, 0)]")
assert np.allclose(gv2, [1, 0, 0], atol=1e-12)

# ------------------------------------------------------------------ Laplacian comparison
# ∇²(ρ²) in cylindrical:
#   (1/ρ)·∂/∂ρ(ρ·2ρ) = (1/ρ)·∂/∂ρ(2ρ²) = 4 = constant
g2 = PowerCylindrical(kz=0, m_azim=0, power=2)   # g = ρ^2
grad_g2 = Cylindrical3D.gradient(g2)
lap_g2 = Cylindrical3D.divergence(grad_g2.components)
r_pts = np.array([[0.5, 0.0, 0.0],
                   [1.0, 1.0, 2.0],
                   [2.0, np.pi, -1.0]])
lap_vals = lap_g2(r_pts)
print("\n=== Laplacian of ρ² in Cylindrical3D ===")
print("  ∇²(ρ²) at sample points =", lap_vals, " [expected 4.0]")
assert np.allclose(lap_vals, 4.0, atol=1e-10)

# Same check in Cartesian: ρ² = x² + y² → ∇² = 4 (in 2D; z-part contributes 0)
# Use PowFunc sum for x^2+y^2 — but easier to check numerically with Exp3D Laplacian
print("\n=== coord_sys carried by Funcs3D ===")
print("  Exp3D.coord_sys name  :", f_cart.coord_sys.name)
print("  PowerCylindrical.coord_sys name:", g2.coord_sys.name)
