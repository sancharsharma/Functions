"""Test 3: integration — fundamental theorem of calculus and the numerical fallbacks.

Ties together the analytic `antiderivative`, the `NotImplementedError`→`numerical_integrate`
fallback in `Funcs1D.integrate`, and `SumOfExps.integrate`'s special k=0 term handling.
"""
import numpy as np
import pytest

import Functions as F

A, B = 0.5, 2.0


@pytest.mark.parametrize("f", [
	F.PolyFunc([1.0, -2.0, 3.0]),
	F.ExpFunc(k=1.2, ampl=2.0, shift=0.3),
	F.SumOfExps([2.0, -1.0], [0.7, 1.3], shifts=[0.1, -0.2]),
], ids=lambda f: type(f).__name__)
def test_definite_integral_matches_antiderivative_and_quad(f):
	definite = f.integrate(A, B)
	# FTC: definite integral equals antiderivative evaluated at the endpoints
	Fanti = f.antiderivative()
	ftc = Fanti(B, ignore_domain=True) - Fanti(A, ignore_domain=True)
	assert definite == pytest.approx(ftc, rel=1e-12, abs=1e-12)
	# ...and an independent numerical reference (scipy quad)
	assert definite == pytest.approx(f.numerical_integrate(A, B), rel=1e-9, abs=1e-9)


def test_powfunc_minus_one_falls_back_to_numerical():
	# 1/x has no PowFunc antiderivative → integrate(a, b) must hit the numerical fallback.
	pw = F.PowFunc(power=-1, ampl=1.0)
	with pytest.raises(NotImplementedError):
		pw.antiderivative()
	assert pw.integrate(1.0, 2.0) == pytest.approx(np.log(2.0), rel=1e-7)


def test_sumofexps_with_constant_term_definite_integral():
	# A k=0 term makes antiderivative() raise; integrate(a, b) uses the term-wise branch
	# (constant term contributes c*(b-a)).
	soe = F.SumOfExps([2.0, 3.0], [0.0, 1.1])
	assert isinstance(soe, F.SumOfExps)
	with pytest.raises(NotImplementedError):
		soe.antiderivative()
	expected = 2.0 * (B - A) + (3.0 / 1.1) * (np.exp(1.1 * B) - np.exp(1.1 * A))
	assert soe.integrate(A, B) == pytest.approx(expected, rel=1e-12)
