"""Test 4: analytic Laplacian vs numerical Laplacian, especially in curved coordinates.

The highest-value test: `laplacian()` drives the whole gradient→divergence→laplacian
pipeline, including `_gradient_component` absorbing 1/rho via Bessel recurrences and the
scale-factor bookkeeping in `divergence`. Cross-checked against `numerical_laplacian`, an
independent finite-difference implementation of the Laplace-Beltrami operator.
"""
import numpy as np
import pytest

import Functions as F
from helpers import PTS_2D, PTS_3D, max_abs_err

# numerical_laplacian is a second-difference of a flux at eps=1e-5 → error ~1e-4 worst case.
FD_ATOL = 1e-3


@pytest.mark.parametrize("f", [
	F.Cylindrical(kz=0.7, m_azim=2, bessel=["J", 2, 1.1], ampl=1.0),
	F.Cylindrical(kz=0.5, m_azim=1, bessel=["J", 1, 0.9], ampl=1.0),
	F.Cylindrical(kz=0.0, m_azim=0, bessel=["J", 0, 1.2], ampl=1.0),
	F.Cylindrical(kz=0.5, m_azim=1, bessel=["I", 1, 0.9], ampl=1.0),
	F.PowerCylindrical(kz=0.6, m_azim=2, power=3, ampl=1.5),
], ids=lambda f: type(f).__name__)
def test_laplacian_cylindrical(f):
	analytic = f.laplacian(F.Cylindrical3D)(PTS_3D)
	numerical = f.numerical_laplacian(PTS_3D, "cylindrical")
	assert max_abs_err(analytic, numerical) < FD_ATOL


def test_laplacian_cartesian_control():
	# Flat-space control: scale factors are all 1, so this isolates the gradient/divergence
	# machinery from the curved-coordinate handling.
	f = F.Exp3D([0.5, -0.3, 0.8], ampl=2.0)
	analytic = f.laplacian(F.Cartesian3D)(PTS_3D)
	numerical = f.numerical_laplacian(PTS_3D, "cartesian")
	assert max_abs_err(analytic, numerical) < FD_ATOL


@pytest.mark.parametrize("f", [
	F.PolarBessel(m_azim=2, bessel=["J", 2, 1.1], ampl=1.0),
	F.PolarBessel(m_azim=1, bessel=["J", 1, 0.9], ampl=1.0),
	F.PolarBessel(m_azim=0, bessel=["J", 0, 1.2], ampl=1.0),
	F.PolarBessel(m_azim=1, bessel=["I", 1, 0.9], ampl=1.0),
	F.PolarPower(m_azim=2, power=3, ampl=1.5),
], ids=lambda f: type(f).__name__)
def test_laplacian_polar(f):
	analytic = f.laplacian(F.Polar2D)(PTS_2D)
	numerical = f.numerical_laplacian(PTS_2D, "polar")
	assert max_abs_err(analytic, numerical) < FD_ATOL


def test_laplacian_cartesian2d_control():
	# 2D flat-space control (all scale factors 1).
	f = F.Exp2D([0.5, -0.3], ampl=2.0)
	analytic = f.laplacian(F.Cartesian2D)(PTS_2D)
	numerical = f.numerical_laplacian(PTS_2D, "cartesian2d")
	assert max_abs_err(analytic, numerical) < FD_ATOL


def test_polar_bessel_is_helmholtz_eigenfunction():
	# J_m(k·r)·e^{imφ} satisfies ∇²f = −k²·f in 2D (analytically, no finite differences).
	f = F.PolarBessel(m_azim=1, bessel=["J", 1, 0.9], ampl=1.0)
	assert max_abs_err(f.laplacian(F.Polar2D)(PTS_2D), -0.9**2 * f(PTS_2D)) < 1e-9


# A modified-Bessel mode I_m(kρ)·e^{imφ}·e^{ikz} (or with K_m) is a separated solution of
# Laplace's equation ∇²f = 0 in cylindrical coordinates when scale = kz and order = m_azim.
# Reason: the radial operator (1/ρ)∂_ρ(ρ∂_ρ)B_n(sρ) = (n²/ρ² + s²)·B_n(sρ) for modified Bessel
# B_n (using x²B″ + xB′ − (x²+n²)B = 0), while (1/ρ²)∂_φ² = −m²/ρ² and ∂_z² = −kz². Summing gives
# (n²−m²)/ρ² + (s²−kz²), which vanishes identically iff n=±m and s=±kz. The analytic laplacian()
# object does not algebraically collapse to ZeroFunc (the cancellation needs a Bessel recurrence
# that simplify() doesn't apply), so we cross-check the value at sample points — as elsewhere here.
@pytest.mark.parametrize("bessel_name", ["I", "K"])
@pytest.mark.parametrize("kz, m_azim", [(3, 2), (2, 0), (1, 3)])
def test_cylindrical_modified_bessel_is_harmonic(bessel_name, kz, m_azim):
	f = F.Cylindrical(kz=kz, m_azim=m_azim, bessel=[bessel_name, m_azim, kz], ampl=1)
	lap = f.laplacian(F.Cylindrical3D)
	assert max_abs_err(lap(PTS_3D), np.zeros(len(PTS_3D))) < 1e-9


@pytest.mark.parametrize("bessel_name", ["J", "Y"])
def test_cylindrical_ordinary_bessel_is_not_harmonic(bessel_name):
	# Contrast: ordinary Bessel with the same scale=kz, order=m_azim solves Helmholtz (∇²f = −2kz²·f)
	# under the same substitution, so its Laplacian is decidedly non-zero — this guards against the
	# harmonic test passing for a trivial/degenerate reason.
	kz, m_azim = 3, 2
	f = F.Cylindrical(kz=kz, m_azim=m_azim, bessel=[bessel_name, m_azim, kz], ampl=1)
	lap = f.laplacian(F.Cylindrical3D)
	assert max_abs_err(lap(PTS_3D), -2 * kz**2 * f(PTS_3D)) < 1e-9
