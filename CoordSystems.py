import numpy as np
import functools
from collections import deque
from dataclasses import dataclass
from . import Functions_Base as _base


def _coord_transform(func):
	"""Wrap an (N, d) → (N, d) coordinate transform so it also accepts a single (d,) vector."""
	@functools.wraps(func)
	def wrapper(pos):
		pos = np.asarray(pos, dtype=float)
		batch = pos.ndim == 2
		p = pos if batch else pos[np.newaxis, :]
		result = func(p)
		return result if batch else result[0]
	return wrapper


class CoordPoint:

	def __init__(self, pos, coord_sys):
		self.pos = np.asarray(pos, dtype=float)
		self.coord_sys = coord_sys

	def convert_to(self, other_sys):
		if other_sys is self.coord_sys:
			return self
		transforms = _find_conversion_path(self.coord_sys, other_sys)
		if transforms is None:
			raise NotImplementedError(f"No conversion path from '{self.coord_sys.name}' to '{other_sys.name}'")
		pos = self.pos
		for transform in transforms:
			pos = transform(pos)
		return CoordPoint(pos, other_sys)

	def __repr__(self):
		return f"CoordPoint({self.pos}, {self.coord_sys.name})"


@dataclass(eq=False)
class CoordSystem:
	"""Pure metric/geometry data for a coordinate system: coordinate names and inverse scale
	factors. Conversions between systems are not stored here — they are directed edges in a
	conversion graph (see `register_conversion` / `_find_conversion_path`); Cartesian is just
	one node like any other. The differential operators (gradient, divergence, laplacian) live
	on the functions themselves (FuncBase) and read this data. Equality is identity-based
	(eq=False) — the module relies on `is` comparisons."""

	name: str
	coords: list
	inv_scale_factors: list

	def __repr__(self):
		return f"CoordSystem({self.name!r}, coords={self.coords})"

	def point(self, pos):
		return CoordPoint(pos, self)


@_coord_transform
def _cyl_to_cart(pos):
	rho, phi, z = pos[:, 0], pos[:, 1], pos[:, 2]
	return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


@_coord_transform
def _cart_to_cyl(pos):
	x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
	return np.stack([np.sqrt(x**2 + y**2), np.arctan2(y, x), z], axis=1)


@_coord_transform
def _polar_to_cart(pos):
	r, phi = pos[:, 0], pos[:, 1]
	return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)


@_coord_transform
def _cart_to_polar(pos):
	x, y = pos[:, 0], pos[:, 1]
	return np.stack([np.sqrt(x**2 + y**2), np.arctan2(y, x)], axis=1)


@_coord_transform
def _sph_to_cart(pos):
	r, theta, phi = pos[:, 0], pos[:, 1], pos[:, 2]
	st = np.sin(theta)
	return np.stack([r*st*np.cos(phi), r*st*np.sin(phi), r*np.cos(theta)], axis=1)


@_coord_transform
def _cart_to_sph(pos):
	x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arccos(np.clip(z / r, -1.0, 1.0))
	phi = np.arctan2(y, x)
	return np.stack([r, theta, phi], axis=1)


@_coord_transform
def _cyl_to_sph(pos):
	# cylindrical (rho, phi, z) → spherical (r, theta, phi), direct (no Cartesian round-trip).
	rho, phi, z = pos[:, 0], pos[:, 1], pos[:, 2]
	return np.stack([np.sqrt(rho**2 + z**2), np.arctan2(rho, z), phi], axis=1)


@_coord_transform
def _sph_to_cyl(pos):
	# spherical (r, theta, phi) → cylindrical (rho, phi, z), direct (no Cartesian round-trip).
	r, theta, phi = pos[:, 0], pos[:, 1], pos[:, 2]
	return np.stack([r * np.sin(theta), phi, r * np.cos(theta)], axis=1)


Cartesian2D = CoordSystem(
	name='cartesian2d',
	coords=['x', 'y'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=2), _base.ConstFunc(1, input_dim=2)],
)

Cartesian3D = CoordSystem(
	name='cartesian3d',
	coords=['x', 'y', 'z'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=3), _base.ConstFunc(1, input_dim=3), _base.ConstFunc(1, input_dim=3)],
)

Cylindrical3D = CoordSystem(
	name='cylindrical',
	coords=['rho', 'phi', 'z'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=3), _base.CoordPow(input_dim=3, coord_index=0, power=-1, coord_name='rho'), _base.ConstFunc(1, input_dim=3)],
)

Polar2D = CoordSystem(
	name='polar',
	coords=['r', 'phi'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=2), _base.CoordPow(input_dim=2, coord_index=0, power=-1, coord_name='r')],
)

Cylindrical2D = CoordSystem(
	name='cylindrical2d',
	coords=['rho', 'phi'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=2), _base.CoordPow(input_dim=2, coord_index=0, power=-1, coord_name='rho')],
)

Polar3D = CoordSystem(
	name='spherical',
	coords=['r', 'theta', 'phi'],
	inv_scale_factors=[
		_base.ConstFunc(1, input_dim=3),
		_base.CoordPow(input_dim=3, coord_index=0, power=-1, coord_name='r'),
		_base.CoordPow(input_dim=3, coord_index=0, power=-1, coord_name='r') * _base.TrigCoord('sin', input_dim=3, coord_index=1, coord_name='theta').reciprocal(),
	],
)


# Conversion graph: directed edges between coordinate systems, partitioned by dimension so 2D
# and 3D systems live in separate graphs. `convert_to` finds the shortest path (fewest hops)
# via BFS, so a directly-registered edge wins over a multi-hop route. Register each direction
# as its own edge — inverses are not assumed.
_CONVERSIONS = {}  # dim -> {src_sys: {dst_sys: transform_fn}}


def register_conversion(src, dst, transform):
	"""Register a directed conversion `src → dst`. `transform` maps an (N, d) / (d,) array of
	`src` coordinates to `dst` coordinates. Only connects same-dimension systems."""
	if len(src.coords) != len(dst.coords):
		raise ValueError(f"cannot register conversion between {len(src.coords)}D '{src.name}' and {len(dst.coords)}D '{dst.name}'")
	_CONVERSIONS.setdefault(len(src.coords), {}).setdefault(src, {})[dst] = transform
	_find_conversion_path.cache_clear()


@functools.lru_cache(maxsize=None)
def _find_conversion_path(src, dst):
	"""BFS over the conversion graph for `src`'s dimension. Returns a tuple of transform
	functions to apply in order, or None if `dst` is unreachable."""
	graph = _CONVERSIONS.get(len(src.coords), {})
	came_from = {src: None}  # node -> (parent, transform used to reach it)
	queue = deque([src])
	while queue:
		node = queue.popleft()
		if node is dst:
			break
		for neighbour, transform in graph.get(node, {}).items():
			if neighbour not in came_from:
				came_from[neighbour] = (node, transform)
				queue.append(neighbour)
	if dst not in came_from:
		return None
	transforms = []
	node = dst
	while came_from[node] is not None:
		parent, transform = came_from[node]
		transforms.append(transform)
		node = parent
	return tuple(reversed(transforms))


register_conversion(Cylindrical3D, Cartesian3D, _cyl_to_cart)
register_conversion(Cartesian3D, Cylindrical3D, _cart_to_cyl)
register_conversion(Polar3D, Cartesian3D, _sph_to_cart)
register_conversion(Cartesian3D, Polar3D, _cart_to_sph)
register_conversion(Cylindrical3D, Polar3D, _cyl_to_sph)   # direct, no Cartesian round-trip
register_conversion(Polar3D, Cylindrical3D, _sph_to_cyl)   # direct, no Cartesian round-trip
# 2D systems (used by Functions_2D and for point conversions)
register_conversion(Polar2D, Cartesian2D, _polar_to_cart)
register_conversion(Cartesian2D, Polar2D, _cart_to_polar)
register_conversion(Cylindrical2D, Cartesian2D, _polar_to_cart)
register_conversion(Cartesian2D, Cylindrical2D, _cart_to_polar)

_COORD_REGISTRY = {
	'cartesian2d':  Cartesian2D,
	'cartesian3d':  Cartesian3D,
	'cartesian':    Cartesian3D,
	'cylindrical':  Cylindrical3D,
	'cylindrical2d': Cylindrical2D,
	'polar':        Polar2D,
	'spherical':    Polar3D,
}
