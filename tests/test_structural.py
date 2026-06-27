"""Tests 7 & 8: __new__ collapse rules, operator identities, __eq__/simplify, guard rails.

These pin down the non-obvious invariants documented in CLAUDE.md that silently affect
every downstream construction.
"""
import numpy as np
import pytest

import Functions as F
from helpers import GRID_1D, max_abs_err
from Functions.Functions_Base import _SENTINEL  # internal sentinel, exercised on purpose


# ------------------------------------------------------------------------------- test 7
def test_new_collapse_rules():
	assert isinstance(F.ConstFunc(0, input_dim=1), F.ZeroFunc)
	assert isinstance(F.PowFunc(power=2, ampl=0), F.ZeroFunc)
	assert isinstance(F.ExpFunc(k=1.0, ampl=0), F.ZeroFunc)
	assert isinstance(F.ExpFunc(k=0, ampl=3.0), F.ConstFunc)
	assert isinstance(F.SumOfExps([2.0], [1.0]), F.ExpFunc)          # single term collapses
	assert isinstance(F.CoordPow(input_dim=3, coord_index=0, power=0), F.ConstFunc)
	# 2D leaf classes collapse to ZeroFunc at zero amplitude, like their 3D analogues.
	assert isinstance(F.Exp2D([0.5, -0.3], ampl=0), F.ZeroFunc)
	assert isinstance(F.PolarPower(m_azim=2, power=3, ampl=0), F.ZeroFunc)
	assert isinstance(F.PolarBessel(m_azim=2, bessel=["J", 2, 1.1], ampl=0), F.ZeroFunc)


def test_ratio_collapse_rules():
	num = F.PolyFunc([1.0, 2.0])
	# ConstFunc denominator → scaled numerator, not a RatioOfFuncs
	ratio = num / F.ConstFunc(4.0, input_dim=1)
	assert not isinstance(ratio, F.RatioOfFuncs)
	assert max_abs_err(ratio(GRID_1D), num(GRID_1D) / 4.0) < 1e-12
	# ZeroFunc numerator → ZeroFunc
	assert isinstance(F.ZeroFunc(input_dim=1) / num, F.ZeroFunc)


def test_zero_multiplication_preserves_output_dim():
	vec = F.ConstFunc([1.0, 2.0, 3.0], input_dim=3)   # output_dim 3
	z = F.ZeroFunc(input_dim=3)
	prod = vec * z
	assert isinstance(prod, F.ZeroFunc)
	assert prod.output_dim == 3


def test_scalar_field_times_vector_field():
	# A scalar field (output_dim 1) times a vector field (output_dim k) must yield an
	# output_dim k function that broadcasts component-wise: result[i, j] = scalar(x_i)·vec_j(x_i).
	# (Regression: ProdOfFuncs._eval used to broadcast a (N,) scalar against an (N, k) vector
	# along the sample axis, giving wrong values.)
	x = np.linspace(0.5, 2.0, 5)
	scalar = F.PolyFunc([0.0, 1.0])                   # f(x) = x, output_dim 1
	vec = F.ConstFunc([1.0, 2.0, 3.0], input_dim=1)   # output_dim 3
	expected = x[:, None] * np.array([1.0, 2.0, 3.0])
	for prod in (scalar * vec, vec * scalar):         # commutative
		assert prod.output_dim == 3
		assert max_abs_err(prod(x), expected) < 1e-12
	# a single input returns a length-3 vector, not a scalar
	assert max_abs_err((scalar * vec)(2.0), 2.0 * np.array([1.0, 2.0, 3.0])) < 1e-12

	# Same property when the vector field is a genuine VecFunc (varying components).
	vfield = F.VecFunc([F.PolyFunc([0.0, 1.0]), F.PolyFunc([1.0, 0.0, 1.0])])  # [x, 1 + x^2]
	prod = scalar * vfield
	assert prod.output_dim == 2
	exp_vfield = np.stack([x * x, x * (1.0 + x**2)], axis=1)
	assert max_abs_err(prod(x), exp_vfield) < 1e-12


def test_incompatible_vector_dims_raise_on_multiply():
	# Two vector fields of different output_dim cannot be multiplied (no broadcasting rule).
	a = F.ConstFunc([1.0, 2.0], input_dim=1)       # output_dim 2
	b = F.ConstFunc([1.0, 2.0, 3.0], input_dim=1)  # output_dim 3
	with pytest.raises(ValueError):
		a * b


def test_combinators_absorb_zerofunc():
	# Multiplying a SumOfFuncs or a ProdOfFuncs by a ZeroFunc must collapse to a ZeroFunc
	# (zero absorption lives in FuncBase.__mul__; the combinators defer to it rather than
	# re-implementing or, in ProdOfFuncs' case, swallowing the zero as an extra factor).
	f = F.PolyFunc([1.0, 2.0])
	g = F.ExpFunc(k=0.5, ampl=1.5)
	z = F.ZeroFunc(input_dim=1)
	sof = f + g
	pof = f * g
	assert isinstance(sof, F.SumOfFuncs) and isinstance(pof, F.ProdOfFuncs)
	assert isinstance(sof * z, F.ZeroFunc)
	assert isinstance(pof * z, F.ZeroFunc)


def test_sumoffuncs_scalar_distributes():
	# The one case SumOfFuncs.__mul__ still specialises: a scalar distributes over the terms.
	f = F.PolyFunc([1.0, 2.0])
	g = F.ExpFunc(k=0.5, ampl=1.5)
	scaled = 3.0 * (f + g)
	assert isinstance(scaled, F.SumOfFuncs)
	assert max_abs_err(scaled(GRID_1D), 3.0 * (f + g)(GRID_1D)) < 1e-12


def test_operator_identities():
	f = F.PolyFunc([1.0, 2.0, 3.0])
	assert (f + 0) is f                       # additive identity (base __add__ path)
	# multiplicative identity: leaf __mul__ folds scalars, so test a base-__mul__ class
	cp = F.CoordPow(input_dim=3, coord_index=0, power=2, coord_name="rho")
	assert (cp * 1) is cp
	with pytest.raises(ZeroDivisionError):
		1 / F.ZeroFunc(input_dim=1)


# ------------------------------------------------------------------------------- test 8
def test_eq_is_value_based_via_simplify():
	f = F.PolyFunc([1.0, 2.0, 3.0])
	g = F.ExpFunc(k=0.5, ampl=1.5)
	assert (f + g) - g == f
	assert 2 * f - f == f
	assert not (f == g)


def test_sumofexps_simplify_merges_terms():
	# Two terms with identical exponents must merge into one, staying numerically equal.
	soe = F.SumOfExps([2.0, 3.0], [1.1, 1.1])
	simplified = soe.simplify()
	assert isinstance(simplified, F.ExpFunc)        # merged 2 → 1 term
	assert max_abs_err(soe(GRID_1D), simplified(GRID_1D)) < 1e-12


def test_expfunc_same_k_addition_fuses():
	a = F.ExpFunc(k=1.3, ampl=2.0)
	b = F.ExpFunc(k=1.3, ampl=-0.5)
	fused = a + b
	assert isinstance(fused, F.ExpFunc)
	assert max_abs_err(fused(GRID_1D), a(GRID_1D) + b(GRID_1D)) < 1e-12


def test_internal_constructor_guards():
	f = F.PolyFunc([1.0, 2.0])
	g = F.ExpFunc(k=0.5)
	with pytest.raises(TypeError):
		F.SumOfFuncs([f, g])               # missing _key=_SENTINEL
	with pytest.raises(TypeError):
		F.ProdOfFuncs([f, g])
	# the sentinel path is the supported internal route
	assert isinstance(F.SumOfFuncs([f, g], _key=_SENTINEL), F.SumOfFuncs)


def test_coord_sys_is_class_level_constant():
	cyl = F.Cylindrical(kz=0.7, m_azim=2, bessel=["J", 2, 1.1], ampl=1.0)
	with pytest.raises(AttributeError):
		cyl.coord_sys = F.Cartesian3D


def test_gradient_on_bare_combinator_requires_coord_sys():
	# A SumOfFuncs has no coord_sys, so gradient() without one must raise ValueError.
	f = F.PowerCylindrical(kz=0.6, m_azim=1, power=2, ampl=1.0)
	g = F.PowerCylindrical(kz=0.6, m_azim=1, power=3, ampl=1.0)
	combo = f + g
	assert isinstance(combo, F.SumOfFuncs)
	with pytest.raises(ValueError):
		combo.gradient()


def test_input_validation_errors():
	f = F.PolyFunc([1.0, 2.0])
	with pytest.raises(ValueError):
		f(1.0 + 2.0j)                      # complex input rejected
	cyl = F.Cylindrical(kz=0.7, m_azim=2, bessel=["J", 2, 1.1], ampl=1.0)
	with pytest.raises(ValueError):
		cyl(np.zeros((4, 2)))              # wrong trailing dimension for a 3D function
