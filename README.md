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

- **Built-in function library** — a collection of 1D functions (`ExpFunc`, `PowFunc`, `PolyFunc`, `SumOfExps`) and 3D functions (`Exp3D`, `Cylindrical`, `PowerCylindrical`).
- **Unified interface** — every function shares a common abstract base (`FuncBase`) with consistent `__call__`, `derivative`, `sympy_output`, and `copy` APIs.
- **Evaluation** — fast NumPy numerical evaluation (`_eval`).
- **Operator overloading** — `+` and `*` produce fused or composed functions; some simplifications are performed automatically (e.g. two exponentials with equal exponents merge into one).
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

See the [`examples/`](examples/) folder for worked scripts covering 1D functions, 3D Cartesian and cylindrical functions, coordinate systems, function composition, and more.

## Module overview

| Module | Contents |
|--------|----------|
| `Functions_Base.py` | `FuncBase` ABC; `ZeroFunc`, `ConstFunc`, `SumOfFuncs`, `ProdOfFuncs`, `ComposedFunc`, `VecFunc`, `CoordPow`, `TrigCoord` |
| `Functions_1D.py` | `Funcs1D`, `ExpFunc`, `PowFunc`, `PolyFunc`, `SumOfExps` |
| `Functions_3D.py` | `Funcs3D`, `Exp3D`, `Cylindrical`, `PowerCylindrical` |
| `CoordSystems.py` | `CoordSystem`, `CoordPoint`; pre-built: `Cartesian2D/3D`, `Cylindrical3D`, `Polar2D/3D`, `Cylindrical2D` |

## Architecture

### `FuncBase` ABC

Every function subclass must implement four methods:

| Method | Contract |
|--------|----------|
| `_eval(pos_arr)` | Takes an `(N, d)` or `(N,)` NumPy array, returns `(N,)` or `(N, k)` |
| `derivative(*args, **kwargs)` | Returns a new `FuncBase` |
| `sympy_output()` | Returns a SymPy expression |
| `copy()` | Returns a deep copy of the same type |

`__call__` applies domain checking → `pos_to_arr` (normalises scalars/vectors to arrays) → `_eval` → unwraps scalar output when a scalar was passed in.

### Operator overloading

`+` and `*` on any two `FuncBase` objects produce `SumOfFuncs` / `ProdOfFuncs`:

- **Simplifications** — leaf classes override `__add__` / `__mul__` where possible (e.g. `0 * func` → `ZeroFunc`).
- **Unhandled combinations** — fall back to `super().__add__` / `super().__mul__` and return `NotImplemented`, so Python can try `__radd__` / `__rmul__`.
- **Direct instantiation** of `SumOfFuncs` / `ProdOfFuncs` raises `TypeError` — always construct them through operators.

### `__new__` simplification

Several classes collapse degenerate cases at construction time:

- `ConstFunc(0)` → `ZeroFunc`
- `CoordPow(..., power=0)` → `ConstFunc(1)`
- `ExpFunc(k, ampl=0)` → `ZeroFunc`
- `SumOfFuncs([])` → `ZeroFunc`; `SumOfFuncs([f])` → `f`

### Gradient hook

`CoordSystem.gradient(f)` checks for a `_gradient_component(coord)` method on `f` before applying the generic `inv_h * f.derivative(coord)` fallback. Implement `_gradient_component` to absorb coordinate-system scale factors analytically (e.g. Bessel recurrences for cylindrical coordinates).

## Adding a new function class

1. Subclass `Funcs1D`, `Funcs3D`, or `FuncBase` directly.
2. Implement `_eval`, `derivative`, `sympy_output`, `copy`.
3. Override `__add__` / `__mul__` for short-circuit simplifications; fall back to `super()` and return `NotImplemented` for unhandled types.
4. Use `__new__` to collapse degenerate cases, e.g. to `ZeroFunc` or `ConstFunc`.
5. For 3D: add `_deriv_{coord}` methods for each supported coordinate; set `coord_sys` as a class attribute; implement `_gradient_component` if analytic scale-factor cancellation is needed.

