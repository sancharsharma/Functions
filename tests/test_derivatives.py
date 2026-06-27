"""Tests 1 & 2: analytic differentiation cross-checked against finite differences.

These pit every analytic `derivative`/`_deriv_*` against `FuncBase.numerical_derivative`
(a central finite difference). A sign error or a wrong Bessel recurrence in
`Cylindrical._deriv_rho` surfaces immediately.
"""
import numpy as np
import pytest

import Functions as F
from helpers import GRID_1D, PTS_2D, PTS_3D, max_abs_err

# Finite-difference comparisons: central difference at eps=1e-5 → error ~1e-6–1e-8.
FD_ATOL = 1e-5


# ----------------------------------------------------------------------------- test 1 (1D)
_1D_FUNCS = [
	F.ExpFunc(k=1.3, ampl=2.0, shift=0.4),
	F.PowFunc(power=2.5, ampl=1.5),
	F.PolyFunc([1.0, -2.0, 0.5, 3.0]),
	F.SumOfExps([2.0, -1.0], [0.7, 1.3], shifts=[0.1, -0.2]),
]


@pytest.mark.parametrize("f", _1D_FUNCS, ids=lambda f: type(f).__name__)
def test_analytic_vs_numerical_derivative_1d(f):
	analytic = f.derivative()(GRID_1D)
	numerical = f.numerical_derivative(GRID_1D, coord_index=0)
	assert max_abs_err(analytic, numerical) < FD_ATOL


# ----------------------------------------------------------------------------- test 1 (3D)
# (function, coord, coord_index) — covers _deriv_x/y/z, _deriv_rho/phi/z across J/Y/I/K.
_3D_CASES = []
for _coord, _idx in [("x", 0), ("y", 1), ("z", 2)]:
	_3D_CASES.append((F.Exp3D([0.5, -0.3, 0.8], ampl=2.0), _coord, _idx))
for _bessel in (["J", 2, 1.1], ["Y", 1, 0.9], ["I", 1, 0.7], ["K", 2, 0.6]):
	for _coord, _idx in [("rho", 0), ("phi", 1), ("z", 2)]:
		_3D_CASES.append((F.Cylindrical(kz=0.7, m_azim=2, bessel=_bessel, ampl=1.0), _coord, _idx))
for _coord, _idx in [("rho", 0), ("phi", 1), ("z", 2)]:
	_3D_CASES.append((F.PowerCylindrical(kz=0.6, m_azim=2, power=3, ampl=1.5), _coord, _idx))


@pytest.mark.parametrize(
	"f, coord, idx", _3D_CASES,
	ids=lambda v: v if isinstance(v, str) else type(v).__name__,
)
def test_analytic_vs_numerical_derivative_3d(f, coord, idx):
	analytic = f.derivative(coord)(PTS_3D)
	numerical = f.numerical_derivative(PTS_3D, coord_index=idx)
	assert max_abs_err(analytic, numerical) < FD_ATOL


# ----------------------------------------------------------------------------- test 1 (2D)
# (function, coord, coord_index) — covers _deriv_x/y (Cartesian) and _deriv_r/phi (polar).
_2D_CASES = []
for _coord, _idx in [("x", 0), ("y", 1)]:
	_2D_CASES.append((F.Exp2D([0.5, -0.3], ampl=2.0), _coord, _idx))
for _bessel in (["J", 2, 1.1], ["Y", 1, 0.9], ["I", 1, 0.7], ["K", 2, 0.6]):
	for _coord, _idx in [("r", 0), ("phi", 1)]:
		_2D_CASES.append((F.PolarBessel(m_azim=2, bessel=_bessel, ampl=1.0), _coord, _idx))
for _coord, _idx in [("r", 0), ("phi", 1)]:
	_2D_CASES.append((F.PolarPower(m_azim=2, power=3, ampl=1.5), _coord, _idx))


@pytest.mark.parametrize(
	"f, coord, idx", _2D_CASES,
	ids=lambda v: v if isinstance(v, str) else type(v).__name__,
)
def test_analytic_vs_numerical_derivative_2d(f, coord, idx):
	analytic = f.derivative(coord)(PTS_2D)
	numerical = f.numerical_derivative(PTS_2D, coord_index=idx)
	assert max_abs_err(analytic, numerical) < FD_ATOL


# ------------------------------------------------------------------------------- test 2
def test_product_rule():
	f = F.PolyFunc([1.0, 2.0])
	g = F.ExpFunc(k=0.7, ampl=1.5)
	h = f * g
	# analytic derivative of the product vs finite differences
	assert max_abs_err(h.derivative()(GRID_1D), h.numerical_derivative(GRID_1D, 0)) < FD_ATOL
	# ...and vs the hand-written Leibniz expression f'g + fg'
	leibniz = f.derivative() * g + f * g.derivative()
	assert max_abs_err(h.derivative()(GRID_1D), leibniz(GRID_1D)) < 1e-9


def test_quotient_rule():
	f = F.PolyFunc([1.0, 2.0])
	g = F.ExpFunc(k=0.7, ampl=1.5)  # never zero on the grid
	q = f / g
	assert max_abs_err(q.derivative()(GRID_1D), q.numerical_derivative(GRID_1D, 0)) < FD_ATOL


def test_chain_rule():
	# exp(x^2): scalar outer ExpFunc, inner PolyFunc
	c = F.ComposedFunc(F.ExpFunc(k=1.0, ampl=1.0), F.PolyFunc([0.0, 0.0, 1.0]))
	assert max_abs_err(c.derivative()(GRID_1D), c.numerical_derivative(GRID_1D, 0)) < FD_ATOL
