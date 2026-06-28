# Coordinate systems & specialised classes

Load this when working with coordinate systems, differential operators in
curvilinear coordinates, `CoordPoint` conversions, `Embed1D`, or `SumOfExps`.

## `CoordSystem` — pure metric data

`CoordSystem` is a `@dataclass(eq=False)` holding only geometry: `name`,
`coords`, `inv_scale_factors`. It owns **no** differential operators — `gradient`
/ `divergence` / `laplacian` are methods on `FuncBase` (see
[architecture.md](architecture.md)) which read this metric data — and **no**
conversion functions (those are graph edges, below). `eq=False` keeps equality
identity-based, since the module compares systems with `is`.

| Method | Description |
|---|---|
| `point(pos)` | Wraps `pos` in a `CoordPoint` tagged with this system |

## Conversion graph

Conversions are **directed edges in a graph**, not attributes of a system —
Cartesian is just one node like any other. `register_conversion(src, dst,
transform)` adds an edge whose `transform` maps an `(N, d)` / `(d,)` array of
`src` coordinates to `dst` coordinates; each direction is registered separately
(inverses are not assumed). The graph is partitioned by dimension
(`_CONVERSIONS[dim]`), so 2D and 3D systems form independent graphs and a
dimension-mismatched `register_conversion` raises `ValueError`.

`_find_conversion_path(src, dst)` does **BFS** (fewest hops) and is
`lru_cache`-memoised; `register_conversion` clears that cache. Because BFS
minimises hops, a directly-registered edge (e.g. `Cylindrical3D ↔ Polar3D`, which
avoids a Cartesian round-trip) is preferred over a multi-hop route; systems
connected only through Cartesian still reach each other transitively. Add direct
edges only where the round-trip is lossy or you have a closed form.

## `CoordPoint`

`CoordPoint(pos, coord_sys)` tags a position array with its coordinate system.
`convert_to(other_sys)` returns `self` if already in `other_sys`, else looks up
the shortest path via `_find_conversion_path` and composes the transforms along
it, raising `NotImplementedError` if no path exists.

## Pre-built coordinate systems

| Name | Coords | Notes |
|---|---|---|
| `Cartesian2D` | `x, y` | trivial scale factors |
| `Cartesian3D` | `x, y, z` | trivial scale factors |
| `Cylindrical3D` | `rho, phi, z` | `inv_h_phi = 1/rho` |
| `Cylindrical2D` | `rho, phi` | 2D polar with `rho` naming |
| `Polar2D` | `r, phi` | 2D polar with `r` naming |
| `Polar3D` | `r, theta, phi` | spherical; internal name is `'spherical'` |

`_COORD_REGISTRY` maps string keys (`'cartesian2d'`, `'cartesian3d'`,
`'cartesian'`, `'cylindrical'`, `'cylindrical2d'`, `'polar'`, `'spherical'`) to
the corresponding objects.

## `Funcs3D.__call__` with points, and numerical Laplacian

`Funcs3D.__call__` also accepts a `CoordPoint` (auto-converts to the function's
`coord_sys` before evaluating) or a **`list` of `CoordPoint` objects**. Each
point is converted individually to the function's `coord_sys` (a no-op when
already in that system), so inputs may be in any system and a batch may freely
mix systems; the converted `.pos` values are stacked into `(N, 3)` and evaluated
once via `super().__call__`.

`Funcs3D.numerical_laplacian(pos, coord_sys, eps=1e-5)` computes the Laplacian
numerically using a second-order finite-difference stencil in curvilinear
coordinates; accepts `coord_sys` as a string (looked up via `_COORD_REGISTRY`)
or a `CoordSystem` object.

## `Embed1D`

`Embed1D(func, input_dim, coord_index, coord_name=None)` **lifts a 1-D function into `input_dim` dimensions** by evaluating it on one coordinate: `Embed1D(func, n, i)(pos) = func(pos[i])`. The wrapped `func` is always a 1-D function (input_dim=1, e.g. `PowFunc`, `Sin`, `Cos`). It owns the column extraction, the partial-derivative dispatch (`derivative(coord)` returns `clone(func=func.derivative())` when `coord == coord_name` — the embedded variable — else `ZeroFunc`), and the SymPy variable relabelling (`func.sympy_output()` with `x` substituted by the coordinate symbol). The internals (`_func`, `_coord_index`, `_coord_name`) are private. The embedding is linear and identity-preserving, so a zero/constant inner func lifts to `ZeroFunc`/`ConstFunc` (collapsed in `__new__`).

Scale factors are built from it: cylindrical `1/h_phi = 1/ρ` is `Embed1D(PowFunc(power=-1), input_dim=3, coord_index=0, coord_name='rho')`. `reciprocal()` delegates to the inner func, so a power inverts cleanly; `Sin`/`Cos` have no analytic reciprocal (csc/sec were removed), so `Embed1D(Sin(), …).reciprocal()` wraps the base `RatioOfFuncs(ConstFunc(1), sin)` — this is how the spherical `1/(r·sin θ)` scale factor in `Polar3D` is built.

## `SumOfExps`

`SumOfExps(coeffs, exponents, shifts=None)` represents
`Σ c_i * exp(k_i * (x - s_i))`. Supports complex exponents (Fourier modes). Key
methods:
- `norm(low, high)` — analytic `∫|f|²` on `[low, high]`, handles near-zero exponents
- `simplify(exp_threshold, coeff_threshold, t_samples, atol, rtol)` — grid-based
  merging of nearby exponents; `atol`/`rtol` require `t_samples`. Uses a
  single-grid scheme with cell size `exp_threshold/2`, so pairs that straddle a
  cell boundary may not be merged. Also drops terms with
  `|c| < coeff_threshold * max|c|`, and optionally verifies accuracy against
  `t_samples`.
- `__mul__` with another `SumOfExps` — outer product of exponents/coefficients
  with numerically stable shift computation
