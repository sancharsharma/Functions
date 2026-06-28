# Purpose

Numerical functions are typically defined with `lambda pos: ...` or `def func(...): ...`, but these representations are fairly limited when it comes to post-processing. For example,

```python
fun = lambda t: a*exp(b*t) + c*exp(d*t)
```

discards all information about the parameters `{a, b, c, d}`.

This library started as a small collection of simple class definitions, e.g. `SumOfExps` implements `sum(a_n * exp(b_n * t))` exactly while keeping the internal variables intact. Over time, it became clear that useful methods could be attached to these classes: `SumOfExps`, for instance, has a `.norm()` method that uses an analytical formula for norm calculations, and an `__add__` method that defines addition between functions (`fun = fun1 + fun2`). As the number of supported functions grew, I developed a common machinery around them and this package took shape.

**If you're wondering why not just use SymPy for symbolic function representation, the answer is simply speed.** This package is *numerical first*, meaning there is essentially no overhead in swapping your `lambda` functions for these classes. A `SumOfExps` object with a thousand exponentials works fine, but constructing a SymPy equivalent with the same number of exponentials alone took about 2.5 seconds on my computer, and I did not dare to evaluate it.

Anyway, here's the library.

# Functions

A function library for Python. Functions carry domain checks, support operator overloading like addition/multiplication, evaluate numerically via NumPy vectorization, and differentiate themselves analytically to return Functions object.

## Features

- **Built-in function library** — a small collection of 1D functions (`ExpFunc`, `PowFunc`, `PolyFunc`, `SumOfExps`), 2D functions (`Exp2D`, `PolarPower`, `PolarBessel`), and 3D functions (`Exp3D`, `Cylindrical`, `PowerCylindrical`).
- **Unified interface** — every function shares a common base (`FuncBase`) with consistent `__call__`, `derivative`, `sympy_output`, and `clone`/`copy` (copy-with-changes) APIs.
- **Evaluation** — fast NumPy numerical evaluation (`_eval`).
- **Operator overloading** — `+`, `-`, `*`, `/` produce fused or composed functions; some simplifications are performed automatically (e.g. two exponentials with equal exponents merge into one).
- **Calculus & vector operators** — analytic `derivative`, `antiderivative`/`integrate`, and `gradient`/`divergence`/`laplacian` over a coordinate system, each with a numerical cross-check counterpart.
- **Domain checking** — each function carries a callable domain predicate enforced on every call (set to True by default.)
- **Coordinate systems** — a `CoordSystem` class with gradient, divergence, and Laplacian operators; pre-built systems for Cartesian 2D/3D, Cylindrical, and Polar coordinates.

## Installation

### Option 1 — Install directly from GitHub

Make sure you have Python ≥ 3.12 installed. Then open a terminal and run:

```bash
pip install git+https://github.com/sancharsharma/Functions.git
```

This downloads the library directly from GitHub and installs it along with its dependencies (`numpy`, `sympy`, `scipy`). You only need to do this once.

### Option 2 — Download and install manually

If Option 1 fails (e.g. because git is not installed on your computer), follow these steps:

**Step 1 — Download the repository.**
Click the green **Code** button on this GitHub page and choose **Download ZIP**. Unzip the folder somewhere on your computer, for example `C:\Users\you\Functions` on Windows or `~/Functions` on Mac/Linux.

**Step 2 — Open a terminal in that folder.**
On Windows, open the folder in File Explorer, then type `cmd` in the address bar and press Enter. On Mac/Linux, open a terminal and run:
```bash
cd ~/Functions
```

**Step 3 — Install the package.**
Make sure you have Python ≥ 3.12 installed. Then run:
```bash
pip install .
```
This installs the library and its dependencies (`numpy`, `sympy`, `scipy`) automatically. You only need to do this once.

### Using the library

Once installed, open your Python script or Jupyter notebook and import whatever you need:
```python
from Functions.Functions_1D import ExpFunc, SumOfExps
```

## Examples

See the [`examples/`](examples/) folder for worked scripts covering 1D functions, 2D Cartesian and polar functions, 3D Cartesian and cylindrical functions, coordinate systems, function composition, and more.

## Tests

The [`tests/`](tests/) folder holds a [pytest](https://pytest.org) suite. Most tests are *cross-checks*: they pit the analytic code paths against independent numerical references (finite-difference derivatives and Laplacians, `scipy.integrate.quad`), so a sign error or a wrong recurrence is caught immediately. The rest pin down structural invariants (operator simplifications, domain/error handling).

Install the optional test dependency and run the suite:

```bash
pip install -e ".[test]"   # installs pytest alongside the package
python3 -m pytest tests/
```

## Module overview

| Module | Contents |
|--------|----------|
| `Functions_Base/` | `FuncBase` base class (`func_base.py`); `ZeroFunc`, `ConstFunc`, `Embed1D` (`leaves.py`); `SumOfFuncs`, `ProdOfFuncs`, `RatioOfFuncs`, `ComposedFunc`, `VecFunc` (`combinators.py`) |
| `Functions_1D.py` | `Funcs1D`, `ExpFunc`, `PowFunc`, `Sin`, `Cos`, `PolyFunc`, `SumOfExps` |
| `Functions_2D.py` | `Funcs2D`, `Exp2D`, `PolarPower`, `PolarBessel` |
| `Functions_3D.py` | `Funcs3D`, `Exp3D`, `Cylindrical`, `PowerCylindrical` |
| `CoordSystems.py` | `CoordSystem`, `CoordPoint`; pre-built: `Cartesian2D/3D`, `Cylindrical3D`, `Polar2D/3D`, `Cylindrical2D` |
| `tests/` | pytest suite: analytic-vs-numerical cross-checks plus structural-invariant tests |

## Architecture

### `FuncBase` base class

`FuncBase` is a plain base class (not an `abc.ABC`): the only method a subclass genuinely has to implement is `_eval`. Everything else is optional and only needed for the operations a given function supports — the unimplemented ones simply raise `NotImplementedError` if called.

| Method | Role | Required? |
|--------|------|-----------|
| `_eval(pos_arr)` | Takes an `(N, d)` or `(N,)` NumPy array, returns `(N,)` or `(N, k)` | Yes |
| `derivative(*args, **kwargs)` | Returns a new `FuncBase` | For differentiation |
| `antiderivative()` | Returns a new `FuncBase` | For integration |
| `sympy_output()` | Returns a SymPy expression | For symbolic export |
| `self.parameters` (set in `__init__`) | Constructor kwargs; drives the inherited `clone`/`copy` | For reconstruction |

`copy()` is **not** something a subclass implements: it is a base-class no-arg alias for `clone()`, the single copy-with-changes customization point (`f.clone(ampl=5)`). Leaf classes feed `clone` by setting `self.parameters`; the combinators hold sub-functions instead and override `clone` directly. Never override `copy`.

`__call__` runs `pos_to_arr` (normalises scalars/vectors to arrays) → domain check → `_eval` → unwraps scalar output when a scalar was passed in. Pass `ignore_domain=True` to skip the domain check.

### Operator overloading

`+` and `*` on any two `FuncBase` objects produce `SumOfFuncs` / `ProdOfFuncs`; `/` produces a `RatioOfFuncs` (whose `derivative` applies the quotient rule), and `-` is sugar for `+ (-1) * …`:

- **Simplifications** — leaf classes override `__add__` / `__mul__` where possible (e.g. `ExpFunc + SumOfExps` → `SumOfExps`).
- **Unhandled combinations** — fall back to `super().__add__` / `super().__mul__` and return `NotImplemented`, so Python can try `__radd__` / `__rmul__`.
- **Direct instantiation** of `SumOfFuncs` / `ProdOfFuncs` / `RatioOfFuncs` raises `TypeError` — always construct them through operators.

### `__new__` simplification

Several classes collapse degenerate cases at construction time, e.g.

- `ConstFunc(0)` → `ZeroFunc`
- `PowFunc(power=0)` → `ConstFunc(ampl)`; `Embed1D(ConstFunc(c), …)` → `ConstFunc(c)`
- `ExpFunc(k, ampl=0)` → `ZeroFunc`; `ExpFunc(k=0)` → `ConstFunc`
- single-term `SumOfExps([c], [k])` → `ExpFunc`
- `SumOfFuncs([f])` → `f`

### Gradient hook

`gradient` is a method on the function itself: `f.gradient(coord_sys=None)` resolves a coordinate system (the explicit argument, else the function's own `coord_sys` class attribute) and, for each component, checks for a `_gradient_component(coord)` method on `f` before applying the generic `inv_h * f.derivative(coord)` fallback. Implement `_gradient_component` to absorb coordinate-system scale factors analytically (e.g. Bessel recurrences for cylindrical coordinates).

## Coordinate systems

What makes the differential operators above work in curvilinear coordinates is that a `CoordSystem` is *pure geometry*: it carries only a name, the coordinate names (`['rho', 'phi', 'z']`), and the inverse scale factors `1/h_i` which are themselves `FuncBase` objects, so e.g. cylindrical's `1/h_phi = 1/rho` is an `Embed1D(PowFunc(power=-1))`. The operators (`gradient`/`divergence`/`laplacian`) live on the functions and read this metric data, so they reduce to the familiar Cartesian forms whenever the scale factors are all 1, and otherwise pick up the right `1/rho`, `1/(r sin θ)`, … factors automatically. Pre-built systems: `Cartesian2D/3D`, `Cylindrical3D`, `Cylindrical2D`, `Polar2D`, `Polar3D` (spherical). Coordinate conversions are supported.

## Adding a new function class

1. Subclass `Funcs1D`, `Funcs2D`, `Funcs3D`, or `FuncBase` directly.
2. Implement `_eval` (the only required method). Add `derivative`, `sympy_output`, and `antiderivative` only for the operations the function should support, and set `self.parameters` (the constructor kwargs) so the inherited `clone`/`copy` works. Do not override `copy`.
3. Override `__add__` / `__mul__` for short-circuit simplifications and fall back to `super()`.
4. Use `__new__` to collapse degenerate cases, e.g. to `ZeroFunc` or `ConstFunc`.
5. For 2D/3D: add `_deriv_{coord}` methods for each supported coordinate if derivative is needed; set `coord_sys` as a class attribute if mixed coordinate systems are desired; implement `_gradient_component` if analytic scale-factor cancellation is needed.

