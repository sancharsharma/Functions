"""Tests 5 & 6: coordinate-system conversions and CoordPoint-aware evaluation.

Test 5 exercises every registered transform plus the BFS shortest-path logic in
`_find_conversion_path` (a directly-registered edge must agree with a composed route).
Test 6 exercises `Funcs3D.__call__`'s point-conversion branch (single and batched).
"""
import numpy as np
import pytest

import Functions as F
from helpers import max_abs_err

RNG = np.random.default_rng(0)


def _random_cart_3d(n=6):
	# spread out, away from the origin/axis so arctan/sqrt are well-conditioned
	pts = RNG.uniform(-2.0, 2.0, size=(n, 3))
	pts[:, :2] += np.sign(pts[:, :2]) * 0.5  # keep x,y away from 0
	return pts


@pytest.mark.parametrize("other", [F.Cylindrical3D, F.Polar3D], ids=lambda s: s.name)
def test_roundtrip_3d(other):
	for p in _random_cart_3d():
		cp = F.CoordPoint(p, F.Cartesian3D)
		back = cp.convert_to(other).convert_to(F.Cartesian3D)
		assert max_abs_err(back.pos, p) < 1e-9


def test_roundtrip_cyl_to_sph():
	for _ in range(6):
		rho = RNG.uniform(0.3, 2.0)
		phi = RNG.uniform(-np.pi + 0.1, np.pi - 0.1)
		z = RNG.uniform(-2.0, 2.0)
		cp = F.CoordPoint([rho, phi, z], F.Cylindrical3D)
		back = cp.convert_to(F.Polar3D).convert_to(F.Cylindrical3D)
		assert max_abs_err(back.pos, [rho, phi, z]) < 1e-9


def test_roundtrip_2d_polar():
	for _ in range(6):
		p = RNG.uniform(-2.0, 2.0, size=2)
		p += np.sign(p) * 0.5
		cp = F.CoordPoint(p, F.Cartesian2D)
		back = cp.convert_to(F.Polar2D).convert_to(F.Cartesian2D)
		assert max_abs_err(back.pos, p) < 1e-9


def test_direct_edge_matches_composed_route():
	# cyl→sph is a *direct* registered edge (_cyl_to_sph); it must agree with the two-hop
	# cyl→cart→sph route. BFS should pick the direct edge, but both must give equal coords.
	for _ in range(6):
		rho = RNG.uniform(0.3, 2.0)
		phi = RNG.uniform(-np.pi + 0.1, np.pi - 0.1)
		z = RNG.uniform(-2.0, 2.0)
		cp = F.CoordPoint([rho, phi, z], F.Cylindrical3D)
		direct = cp.convert_to(F.Polar3D).pos
		two_hop = cp.convert_to(F.Cartesian3D).convert_to(F.Polar3D).pos
		assert max_abs_err(direct, two_hop) < 1e-9


# ------------------------------------------------------------------------------- test 6
def test_funcs3d_eval_at_foreign_coordpoint():
	# Exp3D is a Cartesian function; calling it at a cylindrical CoordPoint must convert
	# first and match evaluation at the manually-converted Cartesian coords.
	e3 = F.Exp3D([0.5, -0.3, 0.8], ampl=2.0)
	cyl_pt = F.CoordPoint([1.0, 0.5, 0.7], F.Cylindrical3D)
	native = cyl_pt.convert_to(F.Cartesian3D).pos
	assert e3(cyl_pt) == pytest.approx(e3(native), rel=1e-12, abs=1e-12)

	# Cylindrical function called at a Cartesian CoordPoint.
	cyl = F.Cylindrical(kz=0.7, m_azim=2, bessel=["J", 2, 1.1], ampl=1.0)
	cart_pt = F.CoordPoint(native, F.Cartesian3D)
	native_cyl = cart_pt.convert_to(F.Cylindrical3D).pos
	assert cyl(cart_pt) == pytest.approx(cyl(native_cyl), rel=1e-12, abs=1e-12)


def test_funcs2d_eval_at_foreign_coordpoint():
	# Exp2D is a Cartesian function; calling it at a polar CoordPoint must convert first.
	e2 = F.Exp2D([0.5, -0.3], ampl=2.0)
	polar_pt = F.CoordPoint([1.0, 0.5], F.Polar2D)
	native = polar_pt.convert_to(F.Cartesian2D).pos
	assert e2(polar_pt) == pytest.approx(e2(native), rel=1e-12, abs=1e-12)

	# PolarBessel is a polar function called at a Cartesian CoordPoint.
	pb = F.PolarBessel(m_azim=2, bessel=["J", 2, 1.1], ampl=1.0)
	cart_pt = F.CoordPoint(native, F.Cartesian2D)
	native_polar = cart_pt.convert_to(F.Polar2D).pos
	assert pb(cart_pt) == pytest.approx(pb(native_polar), rel=1e-12, abs=1e-12)


def test_funcs3d_eval_at_mixed_coordpoint_batch():
	# A list mixing systems must convert element-wise and match individual evaluations.
	e3 = F.Exp3D([0.5, -0.3, 0.8], ampl=2.0)
	pts = [
		F.CoordPoint([1.0, 0.5, 0.7], F.Cylindrical3D),
		F.CoordPoint([0.4, -0.2, 1.1], F.Cartesian3D),
		F.CoordPoint([1.3, 0.9, -0.3], F.Cylindrical3D),
	]
	batch = e3(pts)
	individual = np.array([e3(p) for p in pts])
	assert max_abs_err(batch, individual) < 1e-12
