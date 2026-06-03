"""
SumOfExps: linear combinations of exponentials.

Topics:
  - Construction with real and complex exponents
  - Evaluation and differentiation
  - norm() — computes ∫|f|² over an interval
  - simplify() — merges nearby exponents, drops small terms
  - Arithmetic: adding and multiplying SumOfExps objects
  - Symbolic output
"""

import numpy as np
from Functions.Functions_1D import SumOfExps, ExpFunc

xs = np.linspace(0, 2, 200)

# ------------------------------------------------------------------ Basic construction
# f(x) = 2·e^x + 3·e^{-x}  (hyperbolic-cosine family)
f = SumOfExps(coeffs=[2, 3], exponents=[1, -1])
print("=== SumOfExps: 2·e^x + 3·e^{-x} ===")
print("  f(0)  =", f(0))       # 2 + 3 = 5
print("  f(1)  =", f(1))
print("  sympy =", f.sympy_output())
assert np.isclose(f(0), 5.0)

# ------------------------------------------------------------------ Derivative
# d/dx [c·e^{k·x}] = c·k·e^{k·x}
df = f.derivative()
print("\n=== Derivative ===")
print("  f'(1) =", df(1))
expected = 2 * np.e - 3 / np.e
assert np.isclose(df(1), expected, rtol=1e-10)

# ------------------------------------------------------------------ norm()
# ∫₀¹ |2·e^x + 3·e^{-x}|² dx
n = f.norm(0, 1)
print("\n=== norm(0, 1) ===")
print("  ∫₀¹ |f|² dx =", n)
# Numerical check via quadrature
from scipy import integrate
n_ref, _ = integrate.quad(lambda x: abs(f(x))**2, 0, 1)
assert np.isclose(n, n_ref, rtol=1e-6)

# ------------------------------------------------------------------ Complex exponents (oscillatory)
# g(x) = e^{iπx} + 0.5·e^{2iπx}  — Fourier modes
g = SumOfExps(coeffs=[1.0, 0.5], exponents=[1j * np.pi, 2j * np.pi])
print("\n=== Complex exponents: Fourier modes ===")
print("  g(0)          =", g(0))
print("  |g(0)|        =", abs(g(0)))
print("  ∫₀¹ |g|² dx   =", g.norm(0, 1))   # should be 1^2 + 0.5^2 = 1.25

# ------------------------------------------------------------------ simplify(): merge nearby exponents
# Build a SumOfExps where two exponents are almost identical
eps = 1e-4
h = SumOfExps(coeffs=[1, 1, 1], exponents=[1.0, 1.0 + eps, 2.0])
print("\n=== simplify(): merge close exponents ===")
print(f"  terms before: {len(h.coeffs)}")
h_sim = h.simplify(exp_threshold=0.01)
print(f"  terms after (exp_threshold=0.01): {len(h_sim.coeffs)}")
assert len(h_sim.coeffs) == 2
assert np.allclose(h(xs), h_sim(xs), atol=1e-3)

# simplify(): drop tiny terms
noisy = SumOfExps(coeffs=[10, 1e-9, 1e-9], exponents=[1.0, 2.0, 3.0])
print("\n=== simplify(): drop small coefficients ===")
clean = noisy.simplify(coeff_threshold=1e-8)
print(f"  terms before: {len(noisy.coeffs)}, after: {len(clean.coeffs)}")
assert len(clean.coeffs) == 1

# ------------------------------------------------------------------ Arithmetic
a = SumOfExps(coeffs=[1], exponents=[1.0])
b = SumOfExps(coeffs=[2], exponents=[2.0])
c = a + b
print("\n=== SumOfExps + SumOfExps ===")
print("  (a+b)(1) =", c(1), "expected", np.e + 2*np.e**2)
assert np.isclose(c(1), np.e + 2*np.e**2)

ab = a * b
print("\n=== SumOfExps * SumOfExps ===")
print("  (a·b)(1) =", ab(1), "expected", np.e * 2*np.e**2)
assert np.isclose(ab(1), 2 * np.e**3, rtol=1e-10)

# Adding an ExpFunc to a SumOfExps
from Functions.Functions_1D import ExpFunc
fexp = ExpFunc(k=3, ampl=0.5)
combined = f + fexp
print("\n=== SumOfExps + ExpFunc ===")
print("  type:", type(combined).__name__)
assert isinstance(combined, SumOfExps)
