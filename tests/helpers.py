"""Shared fixtures/utilities for the test suite."""
import numpy as np

# A deterministic 1D sample grid kept away from x=0 (so PowFunc/1-over-x stay finite).
GRID_1D = np.linspace(0.5, 2.0, 9)

# A deterministic batch of 3D points in *cylindrical-safe* coordinates (rho > 0, away from
# the axis) so curved-coordinate scale factors 1/rho stay finite and finite differences in
# numerical_laplacian don't straddle the singular axis.
PTS_3D = np.array([
	[1.0,  0.3,  0.2],
	[1.5,  1.1, -0.4],
	[2.0, -0.5,  0.7],
	[0.8,  2.0,  0.1],
])

# The same idea in 2D (r/rho > 0, away from the polar axis) for 2D function cross-checks.
PTS_2D = np.array([
	[1.0,  0.3],
	[1.5,  1.1],
	[2.0, -0.5],
	[0.8,  2.0],
])


def max_abs_err(a, b):
	"""Max absolute difference, handling real or complex arrays."""
	return float(np.abs(np.asarray(a) - np.asarray(b)).max())
