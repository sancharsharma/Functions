"""
Discrete (integer-domain) sequences: ExpSeq, PolySeq, TabFunc.

Topics:
  - Evaluation at integers and integer arrays
  - Forward and backward finite differences via .derivative()
  - Smart merging: ExpSeq * ExpSeq, ExpSeq + ExpSeq
  - PolySeq arithmetic
  - TabFunc as a sparse lookup table
  - Symbolic output for all three types
"""

import numpy as np
from Functions.Functions_Discrete import ExpSeq, PolySeq, TabFunc
from Functions.Functions_Base import ConstFunc, ZeroFunc

ns = np.array([0, 1, 2, 3, 4, 5], dtype=int)

# ------------------------------------------------------------------ ExpSeq
# a_n = 2 * 3^n
a = ExpSeq(base=3, ampl=2)
print("=== ExpSeq: 2·3^n ===")
print("  a(ns)   =", a(ns))
assert np.allclose(a(ns), 2 * 3**ns)

# Forward difference: Δ_+ a(n) = a(n+1) - a(n) = 2·3^n·(3-1) = 4·3^n
da_fwd = a.derivative('forward')
print("  Δ+a(ns) =", da_fwd(ns), " [expected 4·3^n]")
assert np.allclose(da_fwd(ns), 4 * 3**ns)

# Backward difference: Δ_- a(n) = a(n) - a(n-1) = 2·3^n·(1-1/3) = 4/3·3^n
da_bwd = a.derivative('backward')
print("  Δ-a(ns) =", da_bwd(ns))
assert np.allclose(da_bwd(ns), 2 * 3**ns * (1 - 1/3))

# Same base → merge on addition
b = ExpSeq(base=3, ampl=5)
c = a + b        # → ExpSeq(base=3, ampl=7)
print("  a + b type:", type(c).__name__, "  ampl =", c.ampl)
assert isinstance(c, ExpSeq) and c.ampl == 7

# ExpSeq * ExpSeq multiplies bases
d = ExpSeq(base=2, ampl=1)
e = a * d        # 2·3^n · 2^n = 2·6^n
print("  2·3^n · 2^n =", e(ns), " type:", type(e).__name__)
assert isinstance(e, ExpSeq) and e.base == 6

# base=1 → ConstFunc
print("  ExpSeq(base=1) is ConstFunc:", isinstance(ExpSeq(base=1, ampl=4), ConstFunc))

print("  sympy:", a.sympy_output())

# ------------------------------------------------------------------ PolySeq
# p(n) = 1 + 2n + 3n²
p = PolySeq([1, 2, 3])
print("\n=== PolySeq: 1 + 2n + 3n² ===")
print("  p(ns)       =", p(ns))
assert np.allclose(p(ns), 1 + 2*ns + 3*ns**2)

# Forward difference: Δ_+ p(n) = p(n+1) - p(n)
#   = (1 + 2(n+1) + 3(n+1)²) - (1 + 2n + 3n²) = 2 + 6n + 3 = 5 + 6n  (wait, let's compute)
#   = 1+2n+2+3n²+6n+3 - 1-2n-3n² = 5+6n
dp_fwd = p.derivative('forward')
print("  Δ+p(ns)     =", dp_fwd(ns), " [expected 5+6n]")
expected_dp = 5 + 6*ns
assert np.allclose(dp_fwd(ns), expected_dp)

# PolySeq algebra
q = PolySeq([0, 1])      # n
pq = p + q               # 1 + 3n + 3n^2
print("  (p + n)(ns) =", pq(ns))
assert isinstance(pq, PolySeq)

pr = p * q               # n + 2n^2 + 3n^3
print("  (p * n)(ns) =", pr(ns))
assert np.allclose(pr(ns), ns*(1 + 2*ns + 3*ns**2))

print("  sympy:", p.sympy_output())

# ------------------------------------------------------------------ TabFunc
# Fibonacci-like table: {0:1, 1:1, 2:2, 3:3, 4:5, 5:8}
fib = {0: 1, 1: 1, 2: 2, 3: 3, 4: 5, 5: 8}
t = TabFunc(fib)
print("\n=== TabFunc (Fibonacci values) ===")
print("  t(ns)  =", t(ns))
assert np.allclose(t(ns), [fib[n] for n in ns])

# Default for missing keys
print("  t(10)  =", t(10), " [default 0]")
assert t(10) == 0.0

# Forward difference
dt = t.derivative('forward')   # Δ_+ t(n) = t(n+1) - t(n)
print("  Δ+t(ns) =", dt(ns))
expected_dt = np.array([t(int(n+1)) - t(int(n)) for n in ns])
assert np.allclose(dt(ns), expected_dt)

# Array-based construction (indexed from 0)
arr_t = TabFunc([10, 20, 30, 0, 40])
print("  array TabFunc(ns) =", arr_t(ns))
assert arr_t(0) == 10 and arr_t(3) == 0

# Addition of two TabFuncs
t2 = TabFunc({0: 10, 2: 5})
combined = t + t2
print("  t + t2 at n=0:", combined(0), " [expected", fib[0]+10, "]")
assert combined(0) == fib[0] + 10

print("  sympy:", t.sympy_output())
