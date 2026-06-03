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
