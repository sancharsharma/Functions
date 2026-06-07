"""
1D functions: PolyFunc, ExpFunc, PowFunc.

Topics:
  - Construction and evaluation
  - Smart same-type merging on +
  - Derivatives and derivative_n
  - Symbolic output via sympy_output()
"""

import numpy as np
from Functions.Functions_1D import ExpFunc, PowFunc, PolyFunc

xs = np.array([0.0, 0.5, 1.0, 2.0])

# ------------------------------------------------------------------ PolyFunc
# coeffs in ascending power order: 1 - 3x^2 + 2x^3
p = PolyFunc([1, 0, -3, 2])
print("=== PolyFunc: 1 - 3x^2 + 2x^3 ===")
print("  p(xs)        =", p(xs))
print("  p'(x)        =", p.derivative().sympy_output())
print("  p''(x)       =", p.derivative_n(2).sympy_output())
print("  sympy_output =", p.sympy_output())

# Polynomial arithmetic — adding two PolyFunc fuses coefficients
q = PolyFunc([0, 1])             # x
pq = p + q
print("  (p + x)(xs)  =", pq(xs))
assert isinstance(pq, PolyFunc)

# Scalar multiply
print("  3·p(xs)      =", (3 * p)(xs))

# ------------------------------------------------------------------ ExpFunc
# 3 * exp(2 * (x - 1))
f = ExpFunc(k=2, ampl=3, shift=1)
print("\n=== ExpFunc: 3·exp(2(x−1)) ===")
print("  f(xs)        =", f(xs))
print("  f'(x)        =", f.derivative().sympy_output())
print("  sympy_output =", f.sympy_output())

# Two ExpFunc with the same k fuse analytically into a single ExpFunc
g = ExpFunc(k=2, ampl=1, shift=0)
h = f + g
print("  f + g is still ExpFunc:", type(h).__name__)
assert isinstance(h, ExpFunc), "same-k ExpFunc should fuse"

# Two ExpFunc with different k produce a SumOfExps
from Functions.Functions_1D import SumOfExps
h2 = f + ExpFunc(k=0.5)
print("  f + ExpFunc(k=0.5) is SumOfExps:", isinstance(h2, SumOfExps))

# ------------------------------------------------------------------ PowFunc
# 2 * x^3
r = PowFunc(power=3, ampl=2)
print("\n=== PowFunc: 2x^3 ===")
print("  r(xs)        =", r(xs))
print("  r'(x)        =", r.derivative().sympy_output())
print("  sympy_output =", r.sympy_output())

# Same-power PowFunc fuse on +
r2 = PowFunc(power=3, ampl=5)
fused = r + r2
print("  2x^3 + 5x^3  =", fused.sympy_output())
assert isinstance(fused, PowFunc) and fused.ampl == 7

# PowFunc * PowFunc combines exponents analytically
r3 = PowFunc(power=2, ampl=3)
prod = r * r3
print("  2x^3 · 3x^2  =", prod.sympy_output())
assert isinstance(prod, PowFunc) and prod.power == 5

# ------------------------------------------------------------------ Mixed arithmetic
# PolyFunc absorbs PowFunc(power=integer) on addition
mixed = p + r   # (1 - 3x^2 + 2x^3) + 2x^3 = 1 - 3x^2 + 4x^3
print("\n=== Mixed: PolyFunc + PowFunc ===")
print("  (p + 2x^3)(xs) =", mixed(xs))
assert isinstance(mixed, PolyFunc)

# ------------------------------------------------------------------ integrate
from Functions.Functions_Base import ZeroFunc, ConstFunc, SumOfFuncs
from Functions.Functions_1D import SumOfExps

print("\n=== integrate (definite) ===")

# ZeroFunc: ∫₀⁵ 0 dx = 0
z = ZeroFunc(input_dim=1)
assert z.integrate(0, 5) == 0, "ZeroFunc.integrate"

# ConstFunc: ∫₁⁴ 3 dx = 9
c = ConstFunc(3.0, input_dim=1)
assert abs(c.integrate(1, 4) - 9.0) < 1e-12, "ConstFunc.integrate"

# ExpFunc: ∫₀¹ 3·exp(2(x−1)) dx = 3/2·(1 − e⁻²)
f_int = ExpFunc(k=2, ampl=3, shift=1)
expected_exp = 3/2 * (np.exp(0.0) - np.exp(-2.0))
assert abs(f_int.integrate(0, 1) - expected_exp) < 1e-10, "ExpFunc.integrate"

# ExpFunc with k=0: ∫₀³ 4 dx = 12  (falls back to numerical_integrate)
f0 = ExpFunc(k=0, ampl=4)
assert abs(f0.integrate(0, 3) - 12.0) < 1e-8, "ExpFunc.integrate k=0"

# PowFunc: ∫₀² 2x³ dx = 2·(2⁴/4) = 8
r_int = PowFunc(power=3, ampl=2)
assert abs(r_int.integrate(0, 2) - 8.0) < 1e-10, "PowFunc.integrate"

# PowFunc power=−1: ∫₁ᵉ 1/x dx = 1  (falls back to numerical_integrate)
r_log = PowFunc(power=-1, ampl=1)
assert abs(r_log.integrate(1, np.e) - 1.0) < 1e-8, "PowFunc.integrate power=-1"

# PolyFunc: ∫₀¹ (1 − 3x² + 2x³) dx = [x − x³ + x⁴/2]₀¹ = 0.5
p_int = PolyFunc([1, 0, -3, 2])
assert abs(p_int.integrate(0, 1) - 0.5) < 1e-10, "PolyFunc.integrate"

# SumOfExps: two-term sum
# Term 1: 3·exp(2(x−1)), integral 0→1 = 3/2·(1 − e⁻²)
# Term 2: 1·exp(−1·x),   integral 0→1 = −1·(e⁻¹ − 1) = 1 − e⁻¹
se = SumOfExps([3.0, 1.0], [2.0, -1.0], shifts=[1.0, 0.0])
expected_se = 3/2*(1.0 - np.exp(-2.0)) + (1.0 - np.exp(-1.0))
assert abs(se.integrate(0, 1) - expected_se) < 1e-10, "SumOfExps.integrate"

# SumOfFuncs: ExpFunc + PowFunc (different types, won't fuse → SumOfFuncs)
sf = f_int + r_int
assert isinstance(sf, SumOfFuncs), "f_int + r_int should be SumOfFuncs"
expected_sf = f_int.integrate(0, 1) + r_int.integrate(0, 1)
assert abs(sf.integrate(0, 1) - expected_sf) < 1e-10, "SumOfFuncs.integrate"

print("  All definite integration tests passed.")

print("\n=== integrate (indefinite / antiderivative) ===")

# ZeroFunc.integrate() → ZeroFunc (antiderivative of 0 is 0)
F_zero = z.integrate()
assert isinstance(F_zero, ZeroFunc), "ZeroFunc antiderivative"
assert F_zero(2.0) == 0, "ZeroFunc antiderivative evaluates to 0"

# ConstFunc.integrate() → c·x  (evaluates as ProdOfFuncs with CoordPow)
F_const = c.integrate()
assert abs(F_const(3.0) - 9.0) < 1e-12, "ConstFunc antiderivative: 3·x at x=3 = 9"

# ExpFunc.integrate() → ExpFunc with ampl/k  (antiderivative of 3·exp(2(x−1)))
F_exp = f_int.integrate()
assert isinstance(F_exp, ExpFunc), "ExpFunc antiderivative is ExpFunc"
assert abs(F_exp.ampl - 3/2) < 1e-12, "ExpFunc antiderivative ampl = ampl/k"
assert abs(f_int.integrate(0, 1) - (F_exp(1.0) - F_exp(0.0))) < 1e-12, "ExpFunc antiderivative matches definite"

# PowFunc.integrate() → PowFunc with power+1  (antiderivative of 2x³ is x⁴/2)
F_pow = r_int.integrate()
assert isinstance(F_pow, PowFunc), "PowFunc antiderivative is PowFunc"
assert F_pow.power == 4 and abs(F_pow.ampl - 0.5) < 1e-12, "PowFunc antiderivative: power=4, ampl=0.5"
assert abs(r_int.integrate(0, 2) - (F_pow(2.0) - F_pow(0.0))) < 1e-12, "PowFunc antiderivative matches definite"

# PolyFunc.integrate() → PolyFunc with antiderivative coefficients
F_poly = p_int.integrate()
assert isinstance(F_poly, PolyFunc), "PolyFunc antiderivative is PolyFunc"
assert abs(p_int.integrate(0, 1) - (F_poly(1.0) - F_poly(0.0))) < 1e-12, "PolyFunc antiderivative matches definite"

# SumOfExps.integrate() → SumOfExps with coeffs/k  (all k≠0)
F_se = se.integrate()
assert isinstance(F_se, SumOfExps), "SumOfExps antiderivative is SumOfExps"
assert abs(se.integrate(0, 1) - (F_se(1.0) - F_se(0.0))) < 1e-10, "SumOfExps antiderivative matches definite"

# SumOfFuncs.integrate() → SumOfFuncs of antiderivatives
F_sf = sf.integrate()
assert abs(sf.integrate(0, 1) - (F_sf(1.0) - F_sf(0.0))) < 1e-10, "SumOfFuncs antiderivative matches definite"

print("  All indefinite integration tests passed.")
