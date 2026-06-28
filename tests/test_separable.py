"""Tests for the separable-function machinery: Bessel1D, the public SeparableFunc combinator,
and the _SeparableMixin Template-Method base used to define separable classes from 1-D factors.

The migrated domain classes (Exp2D/3D, PolarPower/Bessel, Cylindrical, PowerCylindrical) are
already cross-checked end-to-end in test_laplacian.py; here we pin down the new pieces directly
and prove the public builder reconstructs the domain classes value-for-value.
"""
import numpy as np
import scipy.special as spec
import pytest

import Functions as F
from Functions.Functions_Base import _SeparableMixin
from helpers import PTS_2D, PTS_3D, GRID_1D, max_abs_err


# ------------------------------------------------------------------------- Bessel1D
def test_bessel1d_eval_and_derivative():
	b = F.Bessel1D("J", 2, scale=0.7, ampl=1.3)
	assert max_abs_err(b(GRID_1D), 1.3 * spec.jv(2, 0.7 * GRID_1D)) < 1e-14
	# d/dx[J_n(s·x)] = s·(J_{n-1}(s·x) − J_{n+1}(s·x))/2
	expected = 1.3 * 0.7 * (spec.jv(1, 0.7 * GRID_1D) - spec.jv(3, 0.7 * GRID_1D)) / 2
	assert max_abs_err(b.derivative()(GRID_1D), expected) < 1e-14


def test_bessel1d_zero_amplitude_collapses():
	assert isinstance(F.Bessel1D("J", 2, ampl=0), F.ZeroFunc)


def test_bessel1d_unknown_kind_raises():
	with pytest.raises(ValueError):
		F.Bessel1D("Q", 1)


# --------------------------------------------------- public SeparableFunc reconstructs classes
def test_separablefunc_reconstructs_exp3d():
	k = [1.0, 2.0, 0.5]
	sf = F.SeparableFunc([F.ExpFunc(k=1j * ki) for ki in k], coord_sys=F.Cartesian3D, ampl=1.3)
	ref = F.Exp3D(k, ampl=1.3)
	assert max_abs_err(sf(PTS_3D), ref(PTS_3D)) < 1e-12
	for c in ("x", "y", "z"):
		assert max_abs_err(sf.derivative(c)(PTS_3D), ref.derivative(c)(PTS_3D)) < 1e-12
	assert max_abs_err(sf.laplacian()(PTS_3D), ref.laplacian()(PTS_3D)) < 1e-12


def test_separablefunc_reconstructs_polarpower():
	sf = F.SeparableFunc([F.PowFunc(3), F.ExpFunc(k=1j * 2)], coord_sys=F.Polar2D, ampl=1.5)
	ref = F.PolarPower(m_azim=2, power=3, ampl=1.5)
	assert max_abs_err(sf(PTS_2D), ref(PTS_2D)) < 1e-12
	assert max_abs_err(sf.laplacian()(PTS_2D), ref.laplacian()(PTS_2D)) < 1e-12


def test_separablefunc_without_coord_sys():
	# Evaluation and derivative-by-name (default x0, x1, …) work; differential operators raise.
	sf = F.SeparableFunc([F.PowFunc(2), F.ExpFunc(k=1j)])
	pt = [2.0, 1.0]
	assert max_abs_err(sf(pt), 2.0**2 * np.exp(1j * 1.0)) < 1e-12
	assert np.isfinite(sf.derivative("x0")(pt))
	with pytest.raises(ValueError):
		sf.gradient()


def test_separablefunc_factor_count_mismatch_raises():
	with pytest.raises(ValueError):
		F.SeparableFunc([F.PowFunc(2)], coord_sys=F.Cartesian3D)  # 1 factor, 3 coords


def test_separablefunc_scalar_multiplication_folds_into_ampl():
	sf = F.SeparableFunc([F.PowFunc(2), F.ExpFunc(k=1j)], coord_sys=F.Polar2D, ampl=1.0)
	scaled = 2.5 * sf
	assert isinstance(scaled, F.SeparableFunc)
	assert max_abs_err(scaled(PTS_2D), 2.5 * sf(PTS_2D)) < 1e-12


# ------------------------------------- _SeparableMixin as a base for a user-defined class
class _SinCos(_SeparableMixin, F.Funcs2D):
	"""A user-defined separable 2-D class: ampl · sin(kx·x) · cos(ky·y), built from 1-D factors
	via the Template-Method mixin, yet free to carry its own state and methods."""

	coord_sys = F.Cartesian2D

	def __init__(self, kx, ky, ampl=1, domain=lambda pos: True):
		super().__init__(domain=domain)
		self.kx, self.ky, self.ampl = kx, ky, ampl
		self.parameters = {"kx": kx, "ky": ky, "ampl": ampl, "domain": domain}

	def _factors(self):
		return [F.Sin(k=self.kx), F.Cos(k=self.ky)]

	def wavenumber_sq(self):                       # a class-specific method, the whole point
		return self.kx**2 + self.ky**2


def test_user_defined_separable_class():
	f = _SinCos(kx=1.3, ky=0.7, ampl=2.0)
	x, y = PTS_2D[:, 0], PTS_2D[:, 1]
	assert max_abs_err(f(PTS_2D), 2.0 * np.sin(1.3 * x) * np.cos(0.7 * y)) < 1e-12
	# derivative(coord) is derived from the factors (∂_x hits only the sin factor)
	assert max_abs_err(f.derivative("x")(PTS_2D), 2.0 * 1.3 * np.cos(1.3 * x) * np.cos(0.7 * y)) < 1e-12
	# laplacian of a product of trig factors is −(kx²+ky²)·f in Cartesian coords
	assert max_abs_err(f.laplacian()(PTS_2D), -f.wavenumber_sq() * f(PTS_2D)) < 1e-9
	# the custom method and class identity survive
	assert f.wavenumber_sq() == pytest.approx(1.3**2 + 0.7**2)
	assert isinstance(2.0 * f, _SinCos)
