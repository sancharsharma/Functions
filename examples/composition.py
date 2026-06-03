"""
Operator overloading and function composition.

Topics:
  - ComposedFunc and the chain rule
  - SumOfFuncs / ProdOfFuncs from + / *
  - ZeroFunc and ConstFunc absorb automatically
  - simplify() flattens nested sums and products
"""

import numpy as np
from Functions.Functions_Base import ZeroFunc, ConstFunc, ComposedFunc
from Functions.Functions_1D import ExpFunc, PowFunc, PolyFunc

xs = np.array([0.0, 0.5, 1.0, 2.0])

# ------------------------------------------------------------------ ComposedFunc
# exp(x^2): outer = ExpFunc(k=1), inner = PowFunc(power=2)
outer = ExpFunc(k=1)
inner = PowFunc(power=2)
composed = ComposedFunc(outer, inner)
print("=== ComposedFunc: exp(x^2) ===")
print("  composed(xs)        =", composed(xs))
expected = np.exp(xs**2)
assert np.allclose(composed(xs), expected)

# Chain rule: d/dx exp(x^2) = 2x·exp(x^2)
dc = composed.derivative()
print("  d/dx exp(x^2) at xs =", dc(xs))
assert np.allclose(dc(xs), 2 * xs * np.exp(xs**2))
print("  sympy_output        =", composed.sympy_output())

# ------------------------------------------------------------------ SumOfFuncs / ProdOfFuncs
f = PowFunc(power=2, ampl=1)   # x^2
g = ExpFunc(k=1, ampl=1)       # e^x
s = f + g
from Functions.Functions_Base import SumOfFuncs, ProdOfFuncs
print("\n=== SumOfFuncs: x^2 + e^x ===")
print("  (f+g)(xs)    =", s(xs))
assert np.allclose(s(xs), xs**2 + np.exp(xs))
print("  derivative   =", s.derivative().sympy_output())

prod = f * g
print("\n=== ProdOfFuncs: x^2 · e^x ===")
print("  (f*g)(xs)    =", prod(xs))
assert np.allclose(prod(xs), xs**2 * np.exp(xs))
# Product rule: (x^2·e^x)' = 2x·e^x + x^2·e^x
dprod = prod.derivative()
print("  (f*g)'(xs)   =", dprod(xs))
assert np.allclose(dprod(xs), 2*xs*np.exp(xs) + xs**2*np.exp(xs))

# ------------------------------------------------------------------ Absorbing zero and constants
print("\n=== ZeroFunc absorbs ===")
z = ZeroFunc(input_dim=1)
print("  f + ZeroFunc is f:", f + z is f)
print("  f * ZeroFunc is ZeroFunc:", isinstance(f * z, ZeroFunc))

print("\n=== ConstFunc(0) → ZeroFunc ===")
c0 = ConstFunc(0, input_dim=1)
print("  ConstFunc(0) is ZeroFunc:", isinstance(c0, ZeroFunc))

c5 = ConstFunc(5, input_dim=1)
print("  ConstFunc(5) + ConstFunc(3) =", (c5 + ConstFunc(3, input_dim=1))(1.0))

# ------------------------------------------------------------------ simplify()
# Force a 3-term SumOfFuncs by mixing types, then simplify merges same-type terms.
# (x^2 + e^x) + x^2  →  SumOfFuncs([x^2, e^x, x^2])
#  simplify()         →  SumOfFuncs([2x^2, e^x])  (two PowFunc(2) merge)
p_sq = PowFunc(power=2)
e_x  = ExpFunc(k=1)
mixed3 = (p_sq + e_x) + PowFunc(power=2)  # 3-term SumOfFuncs
print("\n=== simplify() merges compatible terms ===")
print("  terms before simplify:", len(mixed3.funcs))   # 3
simplified = mixed3.simplify()
print("  terms after  simplify:", len(simplified.funcs))  # 2
assert len(simplified.funcs) == 2
assert np.allclose(simplified(xs), 2*xs**2 + np.exp(xs))
