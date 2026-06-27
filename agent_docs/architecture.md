# Architecture — FuncBase & combinators

Load this when editing or adding function classes, or when reasoning about how
combination/simplification/dimension rules work.

## FuncBase ABC

Every function subclass must implement:
- `_eval(pos_arr)` — takes an `(N, d)` or `(N,)` NumPy array, returns `(N,)` or `(N, k)`.
- `derivative(*args, **kwargs)` — returns a new `FuncBase`.
- `sympy_output()` — returns a SymPy expression.

The public reconstruction hook is `clone(**overrides)` — a copy-with-changes
(`f.clone(ampl=5)`, cf. `dataclasses.replace`). `copy()` is **not** required and is
defined only on `FuncBase` as `self.clone()` (a no-arg alias). Leaf classes set
`self.parameters` (a dict of their constructor kwargs) in `__init__` and inherit the
default `clone` (`type(self)(**{**self.parameters, **overrides})`), which also powers
concise reconstruction inside `derivative`/`__mul__` (e.g.
`self.clone(ampl=k*self.ampl)`). Combinators (`SumOfFuncs`, `ProdOfFuncs`,
`RatioOfFuncs`, `ComposedFunc`, `VecFunc`) hold sub-functions rather than flat
parameters, so they override `clone` (not `copy`) to deep-copy children by default,
accepting overrides keyed by the constructor argument (`funcs`, `numer`/`denom`,
`outer`/`inner`, `components`).

`__call__` runs domain checking → `pos_to_arr` (normalises scalars/vectors to
arrays and records `single_input`) → `_eval` → unwrap if `single_input`.

## `pos_to_arr` — input normalisation

| `input_dim` | Accepted input | Returns |
|---|---|---|
| `1` | scalar or 1D array | `(1D arr, single_input)` |
| `d > 1` | `(d,)` vector or `(N, d)` array | `((N,d) arr, single_input)` |

Complex inputs raise `ValueError`; wrong shapes raise `ValueError`. For a 1D
function a scalar returns `single_input=True`.

## Operator overloading

`+`, `*`, `-` on any two `FuncBase` objects produce `SumOfFuncs` / `ProdOfFuncs`.
`/` produces `RatioOfFuncs` (or simplifies to a scaled copy when the denominator
is `ConstFunc`).

`__neg__` returns `(-1) * self`; `__sub__` returns `self + (-1) * other`.

Many leaf classes override `__add__` / `__mul__` to fuse same-type pairs
analytically (e.g. two `ExpFunc` with the same `k`, two `PowFunc` with the same
power). Always fall back to `super().__add__` / `super().__mul__` for unhandled
types; return `NotImplemented` so Python can try `__radd__` / `__rmul__`.

Helpers in `Functions_Base` keep these overloads terse:
- `as_scalar(x)` → a Python scalar if `x` is a numeric scalar (incl. a 0-d
  ndarray), else `None`. Never unwraps a `FuncBase`.
- `scalar_factor(x)` → as above but also unwraps a scalar (`output_dim==1`)
  `ConstFunc` to its value (dropping its domain); used by leaf classes to fold a
  scalar multiplier into a parameter.
- `combine_domains(*items)` → an AND of the domains of the given `FuncBase`
  objects (or raw domain callables).

`FuncBase.__add__`/`__mul__` treat numeric `0`/`1` as additive/multiplicative
identities (returning `self`), and `scalar * f == 0` returns `ZeroFunc`. This is
what lets the codebase use the builtins `sum(...)` and `math.prod(...)` directly
(there is no `gen_sum`/`gen_prod`); `sum([])`/`math.prod([])` still give the
`0`/`1` needed by the Leibniz product rule.

`FuncBase.__mul__` splits on a single test — `isinstance(other, FuncBase)`. A
non-`FuncBase` operand is either a numeric scalar (identities / fold into
`ConstFunc`) or a 1D array/list (wrapped in `ConstFunc` → `ProdOfFuncs([ConstFunc(arr),
self])`, raising `ValueError` if its length ≠ `self.output_dim`); anything else
returns `NotImplemented`. A `FuncBase` operand is first run through
`_check_dim_compat([self, other], 'mul')` (so a dimension mismatch raises even when
`other` is a `ZeroFunc`), then `ZeroFunc` absorbs to a `ZeroFunc` of the combined
shape and everything else becomes `ProdOfFuncs([self, other])`.

`__rmul__` (and `__radd__`) are defined **once** on `FuncBase` and delegate to
`__mul__`/`__add__`; subclasses inherit them, so `arr * f` and `f * arr` follow the
same path. Do not re-define `__rmul__` on subclasses — overriding `__mul__` is enough.

`SumOfFuncs` and `ProdOfFuncs` are internal — construct them only through
operators. Direct instantiation raises `TypeError` (protected by `_SENTINEL`).
Same for `RatioOfFuncs`.

`FuncBase.__eq__` checks symbolic equality by simplifying `self - other` and
testing whether the result is a `ZeroFunc`. It coerces the right operand through
subtraction, so `ConstFunc(5) == 5` and `ZeroFunc == 0` are `True`; if the
subtraction fails or yields a non-`FuncBase`, it returns `NotImplemented`.
Because equality is value-based (and instances are mutable), `__hash__ = None` —
the objects are intentionally **unhashable**.

## Division and `RatioOfFuncs`

`f / scalar` returns a scaled copy (`f * (1/scalar)`). `f / g` returns
`RatioOfFuncs(f, g)` unless `g` is `ConstFunc` (simplifies immediately).
`scalar / f` uses `__rtruediv__` → `RatioOfFuncs(ConstFunc(scalar), f)`.

`RatioOfFuncs.__new__` simplifies: `/ZeroFunc` raises `ZeroDivisionError`;
`ZeroFunc / g` returns `ZeroFunc`; `f / ConstFunc` returns `(1/const) * f`.

`RatioOfFuncs.derivative` applies the quotient rule analytically.

`FuncBase.reciprocal()` builds `RatioOfFuncs(ConstFunc(1), self)` (or a vector
`ConstFunc(ones)` for vector output). `ConstFunc.reciprocal()` and
`CoordPow.reciprocal()` return analytic inverses directly.
`ProdOfFuncs.reciprocal()` distributes over factors. `TrigCoord` has no analytic
reciprocal (csc/sec were removed), so `TrigCoord('sin', …).reciprocal()` falls
back to the base `RatioOfFuncs(ConstFunc(1), sin)` — this is how the spherical
`1/(r·sin θ)` scale factor is built in `Polar3D`.

## Dimension compatibility — `_check_dim_compat(funcs, mode)`

All combination sites validate both `input_dim` and `output_dim` through the
single helper `_check_dim_compat(funcs, mode)`, which returns
`(input_dim, output_dim)`.

| `mode` | `output_dim` rule | Used by |
|---|---|---|
| `'sum'` | all must be equal (or `None`) | `ZeroFunc.__add__`, `ConstFunc.__add__`, `SumOfFuncs.__init__`, `VecFunc.__init__`* |
| `'mul'` | scalar (`1` or `None`) defers to the other; equal dims are fine; else raises | `FuncBase.__mul__`, `ZeroFunc.__mul__`, `ConstFunc.__mul__`, `ProdOfFuncs.__init__`, `RatioOfFuncs.__init__` |

\* `VecFunc.__init__` passes `[*components, ConstFunc(1)]` with `mode='sum'` to
enforce that every component is scalar; `VecFunc.output_dim` is then set to
`len(components)` independently.

`RatioOfFuncs` uses `'mul'`, so `scalar / vector` is valid and produces a vector
result. `ProdOfFuncs` is the same: a scalar field (`output_dim 1`) times a vector
field (`output_dim k`) is a vector. Both `ProdOfFuncs._eval` and `RatioOfFuncs._eval`
promote a scalar `(N,)` factor to `(N, 1)` so it broadcasts **component-wise** against
the `(N, k)` vector rather than along the sample axis.

## `__new__` simplification

Several classes simplify on construction in `__new__`:
- `ConstFunc(0)` → `ZeroFunc`
- `CoordPow(..., power=0)` → `ConstFunc(1)`
- `ComposedFunc` with constant inner/outer → `ConstFunc`
- `SumOfFuncs([])` → `ZeroFunc`; `SumOfFuncs([f])` → `f`
- `ProdOfFuncs([])` → `ConstFunc(1)`; `ProdOfFuncs([f])` → `f`
- `RatioOfFuncs(ZeroFunc, g)` → `ZeroFunc`; `/ZeroFunc` → `ZeroDivisionError`; `/ConstFunc` → scaled copy
- `ExpFunc(ampl=0)`, `PowFunc(ampl=0)`, `PolyFunc(all-zero)`, `SumOfExps(all-zero)` → `ZeroFunc`
- `ExpFunc(k=0)` → `ConstFunc(ampl)`
- `SumOfExps` with a single term → `ExpFunc` (which may itself collapse to `ConstFunc`)

## Domain checking

`check_domain(pos, output='error')` accepts three modes:
- `'error'` (default) — raises `ValueError` if any points violate the domain
- `'points'` — returns the list of violating points
- `'binary'` — returns `True` if all points are in the domain

Pass `ignore_domain=True` to `__call__` to skip domain checks (used internally
for finite-difference stencils).

## Numerical & differential methods on `FuncBase`

- `numerical_derivative(pos, coord_index=0, eps=1e-5)` — central finite difference.
- `gradient(coord_sys=None)` — `VecFunc` of `inv_h_i · df/dq_i` (with the
  gradient hook). Requires a **scalar field** (`output_dim == 1`) whose
  `input_dim` matches `len(coord_sys.coords)`.
- `divergence(coord_sys=None)` — divergence of a vector field; requires
  `output_dim == input_dim == len(coord_sys.coords)` (a `VecFunc`, or a
  vector-valued `Funcs3D` with `output_dim=3`, …). Reads components via
  `self[i]`, so the function must support indexing. Scalar fields are rejected.
- `laplacian(coord_sys=None)` — `self.gradient(coord_sys).divergence(coord_sys)`.

The differential operators live **on the function**, not the coordinate system;
they read the metric (`coords`, `inv_scale_factors`) off `coord_sys`.
`coord_sys` defaults to `self.coord_sys` (a class attribute on `Funcs3D`
subclasses); combinators/`VecFunc` have none, so they require it explicitly or
raise `ValueError`.

`numerical_integrate(a, b)` lives on **`Funcs1D`** (not `FuncBase`):
scipy.integrate.quad, 1D scalar functions only.

## Analytic integration

`antiderivative()` is the per-class method that returns an antiderivative
`FuncBase` (or raises `NotImplementedError`). `integrate(a=None, b=None)` is
defined generically on **`FuncBase`** (indefinite → `antiderivative()`; definite
→ `F(b)−F(a)`), so `ZeroFunc`/`ConstFunc`/etc. all support it. Two subclasses
override it: `Funcs1D` adds a `numerical_integrate` fallback when there is no
analytic antiderivative; `SumOfExps` computes the definite integral term-by-term
(so `k=0` terms work).

| Call | Returns | Fallback |
|---|---|---|
| `f.integrate()` | `f.antiderivative()` | — |
| `f.integrate(a, b)` | definite integral scalar | `F(b)−F(a)`; on `Funcs1D`, falls back to `numerical_integrate(a, b)` if there is no antiderivative |

`antiderivative()` is implemented on:

| Class | `antiderivative()` returns | Edge case |
|---|---|---|
| `ZeroFunc` | `ZeroFunc` (copy) | — |
| `ConstFunc` | `const * CoordPow(x)` (i.e. `const·x`) | non-1D (`input_dim≠1` or `output_dim≠1`) raises |
| `ExpFunc` | `ExpFunc(k, ampl/k, shift)` | `k=0` never reaches here (collapses to `ConstFunc` at construction) |
| `PowFunc` | `PowFunc(power+1, ampl/(power+1))` | power=−1 raises; definite falls to numerical |
| `PolyFunc` | `PolyFunc` with antiderivative coefficients `[0, c₀, c₁/2, …]` | — |
| `SumOfExps` | `SumOfExps(coeffs/k, …)` | k=0 terms raise; definite computed analytically regardless |
| `SumOfFuncs` | `SumOfFuncs` of each term's antiderivative | — |

`Funcs3D` defines neither `integrate` nor `antiderivative` (it inherits the
generic `FuncBase.integrate`, which would raise via the missing `antiderivative`).

## `simplify()` on combinators

`SumOfFuncs.simplify(deep=False)` tries pairwise `fi + fj`; if the result is not
a `SumOfFuncs`, it replaces both with the merged term. `deep=True` first
recursively simplifies each sub-function. `ProdOfFuncs.simplify()` does the same
pairwise with `*`. Both delegate the loop to the shared helper
`_fuse_pairwise(funcs, op, container_cls)`.

## `VecFunc`

`VecFunc(components)` stacks scalar functions into a vector-valued function.
Supports:
- `__add__` with another `VecFunc` of the same length (component-wise)
- `__mul__` with a scalar, a 1D array (element-wise by component), another
  `VecFunc` (component-wise), or a scalar `FuncBase`
- `__getitem__(i)` returns the i-th component function
- `derivative(*args, **kwargs)` returns a new `VecFunc` by differentiating each
  component

## Gradient hook

`FuncBase.gradient` calls `self._gradient_component(coord)` for each coordinate.
The base method returns `NotImplemented`, so the default falls back to
`inv_h * self.derivative(coord)`. Override `_gradient_component` on a class to
absorb scale factors analytically (e.g. `Cylindrical._gradient_component('phi')`
uses Bessel recurrences to cancel the `1/ρ` factor without creating a `CoordPow`
product). A hook may **selectively** return `NotImplemented` to use the fallback
for cases it cannot handle: `Cylindrical` does this for `order == 0` (the
recurrence needs `1/(2·order)`), so the `1/ρ · ∂_φ` term is still produced
correctly via `inv_h · derivative`.

`SumOfFuncs._gradient_component(coord)` propagates the hook to each term;
returns `NotImplemented` if any term returns `NotImplemented`.

## 2D/3D derivative dispatch

`Funcs3D.derivative(coord)` looks up `_deriv_{coord}(self)` by name (e.g.
`_deriv_rho`, `_deriv_phi`, `_deriv_z`). Known but unimplemented coordinates
raise `NotImplementedError`; unknown names raise `ValueError`. Known
coordinates: `{'x', 'y', 'z', 'rho', 'phi', 'r', 'theta'}`.

`Funcs2D` is the same machinery at `input_dim=2` (its own `_KNOWN_COORDS =
{'x', 'y', 'rho', 'phi', 'r'}`). The 2D leaf classes parallel the 3D ones:
`Exp2D` (Cartesian2D, like `Exp3D`), `PolarPower` (Polar2D, like
`PowerCylindrical`), and `PolarBessel` (Polar2D, like `Cylindrical` — same
Bessel J/Y/I/K recurrences for `_deriv_r` and the `_gradient_component('phi')`
1/r absorption, minus the z dependence). `numerical_laplacian` is
dimension-generic (it reads `n = len(coord_sys.coords)` off the system), so the
2D and 3D copies are identical aside from the surrounding class.

## Adding a new function class

1. Subclass `Funcs1D`, `Funcs3D`, or `FuncBase` directly. Dimension-specific classes
   go in `Functions_{1,2,3}D.py`. New *core* classes go in the `Functions_Base/`
   package: leaf primitives in `leaves.py`, combinators in `combinators.py`. If such a
   class is referenced by name from `func_base.py`/`leaves.py`, add it to the
   **module-bottom import** there (cross-class refs must resolve at call time, not import
   time, to keep the cycle broken).
2. Implement `_eval`, `derivative`, `sympy_output`; set `self.parameters` in
   `__init__` (constructor kwargs) so the inherited `clone`/`copy` work. If the class
   holds sub-functions rather than flat parameters, override `clone` (never `copy`)
   like the combinators do.
3. Override `__add__` / `__mul__` for same-type fusions; use `scalar_factor` /
   `combine_domains`; always fall back to `super()` and return `NotImplemented`
   for unknowns.
4. Use `__new__` to collapse degenerate cases (zero amplitude, etc.) to
   `ZeroFunc` or `ConstFunc`.
5. For 3D: add `_deriv_{coord}` methods for each supported coordinate; add
   `coord_sys` as a class attribute; implement `_gradient_component` if analytic
   scale-factor cancellation is possible.
