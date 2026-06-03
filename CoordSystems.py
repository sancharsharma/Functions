import numpy as np
from . import Functions_Base as _base


class CoordPoint:

	def __init__(self, pos, coord_sys):
		self.pos = np.asarray(pos, dtype=float)
		self.coord_sys = coord_sys

	def convert_to(self, other_sys):
		if other_sys is self.coord_sys:
			return self
		if self.coord_sys.to_cartesian is None or other_sys.from_cartesian is None:
			raise NotImplementedError(f"No conversion defined from '{self.coord_sys.name}' to '{other_sys.name}'")
		cartesian_pos = self.coord_sys.to_cartesian(self.pos)
		return CoordPoint(other_sys.from_cartesian(cartesian_pos), other_sys)

	def __repr__(self):
		return f"CoordPoint({self.pos}, {self.coord_sys.name})"


class CoordSystem:

	def __init__(self, name, coords, inv_scale_factors, to_cartesian=None, from_cartesian=None):
		self.name = name
		self.coords = coords
		self.inv_scale_factors = inv_scale_factors
		self.to_cartesian = to_cartesian
		self.from_cartesian = from_cartesian

	def point(self, pos):
		return CoordPoint(pos, self)

	def gradient(self, f):
		if f.input_dim is not None and f.input_dim != len(self.coords):
			raise ValueError(f"gradient: function input_dim={f.input_dim} does not match coordinate system dimension {len(self.coords)}")
		if f.output_dim is not None and f.output_dim != 1:
			raise ValueError(f"gradient: function output_dim={f.output_dim} must be 1 (scalar field)")
		components = []
		for coord, inv_h in zip(self.coords, self.inv_scale_factors):
			hook = getattr(f, '_gradient_component', None)
			result = hook(coord) if hook is not None else NotImplemented
			if result is NotImplemented:
				df = f.derivative(coord)
				result = df if (isinstance(inv_h, _base.ConstFunc) and inv_h.const == 1) else inv_h * df
			components.append(result)
		return _base.VecFunc(components)

	def divergence(self, components):
		n = len(self.coords)
		if len(components) != n:
			raise ValueError(f"divergence: expected {n} components, got {len(components)}")
		for i, fi in enumerate(components):
			if fi.input_dim is not None and fi.input_dim != n:
				raise ValueError(f"divergence: component {i} input_dim={fi.input_dim} does not match coordinate system dimension {n}")
			if fi.output_dim is not None and fi.output_dim != 1:
				raise ValueError(f"divergence: component {i} output_dim={fi.output_dim} must be 1")
		scale_factors = [inv_h.reciprocal() for inv_h in self.inv_scale_factors]
		terms = []
		for i, (coord, fi) in enumerate(zip(self.coords, components)):
			others_h = [scale_factors[j] for j in range(n) if j != i and not (isinstance(scale_factors[j], _base.ConstFunc) and scale_factors[j].const == 1)]
			coeff_fi = _base.gen_prod(others_h + [fi])
			terms.append(coeff_fi.derivative(coord))
		total = _base.gen_sum(terms)
		nontrivial_inv = [inv_h for inv_h in self.inv_scale_factors if not (isinstance(inv_h, _base.ConstFunc) and inv_h.const == 1)]
		result = total
		for inv_h in nontrivial_inv:
			result = inv_h * result
		return result

	def laplacian(self, f):
		return self.divergence(self.gradient(f).components)


def _cyl_to_cart(pos):
	pos = np.asarray(pos, dtype=float)
	batch = pos.ndim == 2
	p = pos if batch else pos[np.newaxis, :]
	rho, phi, z = p[:, 0], p[:, 1], p[:, 2]
	cart = np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
	return cart if batch else cart[0]


def _cart_to_cyl(pos):
	pos = np.asarray(pos, dtype=float)
	batch = pos.ndim == 2
	p = pos if batch else pos[np.newaxis, :]
	x, y, z = p[:, 0], p[:, 1], p[:, 2]
	cyl = np.stack([np.sqrt(x**2 + y**2), np.arctan2(y, x), z], axis=1)
	return cyl if batch else cyl[0]


def _polar_to_cart(pos):
	pos = np.asarray(pos, dtype=float)
	batch = pos.ndim == 2
	p = pos if batch else pos[np.newaxis, :]
	r, phi = p[:, 0], p[:, 1]
	cart = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)
	return cart if batch else cart[0]


def _cart_to_polar(pos):
	pos = np.asarray(pos, dtype=float)
	batch = pos.ndim == 2
	p = pos if batch else pos[np.newaxis, :]
	x, y = p[:, 0], p[:, 1]
	polar = np.stack([np.sqrt(x**2 + y**2), np.arctan2(y, x)], axis=1)
	return polar if batch else polar[0]


def _sph_to_cart(pos):
	pos = np.asarray(pos, dtype=float)
	batch = pos.ndim == 2
	p = pos if batch else pos[np.newaxis, :]
	r, theta, phi = p[:, 0], p[:, 1], p[:, 2]
	st = np.sin(theta)
	cart = np.stack([r*st*np.cos(phi), r*st*np.sin(phi), r*np.cos(theta)], axis=1)
	return cart if batch else cart[0]


def _cart_to_sph(pos):
	pos = np.asarray(pos, dtype=float)
	batch = pos.ndim == 2
	p = pos if batch else pos[np.newaxis, :]
	x, y, z = p[:, 0], p[:, 1], p[:, 2]
	r = np.sqrt(x**2 + y**2 + z**2)
	theta = np.arccos(np.clip(z / r, -1.0, 1.0))
	phi = np.arctan2(y, x)
	sph = np.stack([r, theta, phi], axis=1)
	return sph if batch else sph[0]


Cartesian2D = CoordSystem(
	name='cartesian2d',
	coords=['x', 'y'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=2), _base.ConstFunc(1, input_dim=2)],
	to_cartesian=lambda pos: np.asarray(pos, dtype=float),
	from_cartesian=lambda pos: np.asarray(pos, dtype=float),
)

Cartesian3D = CoordSystem(
	name='cartesian3d',
	coords=['x', 'y', 'z'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=3), _base.ConstFunc(1, input_dim=3), _base.ConstFunc(1, input_dim=3)],
	to_cartesian=lambda pos: np.asarray(pos, dtype=float),
	from_cartesian=lambda pos: np.asarray(pos, dtype=float),
)

Cylindrical3D = CoordSystem(
	name='cylindrical',
	coords=['rho', 'phi', 'z'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=3), _base.CoordPow(input_dim=3, coord_index=0, power=-1, coord_name='rho'), _base.ConstFunc(1, input_dim=3)],
	to_cartesian=_cyl_to_cart,
	from_cartesian=_cart_to_cyl,
)

Polar2D = CoordSystem(
	name='polar',
	coords=['r', 'phi'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=2), _base.CoordPow(input_dim=2, coord_index=0, power=-1, coord_name='r')],
	to_cartesian=_polar_to_cart,
	from_cartesian=_cart_to_polar,
)

Cylindrical2D = CoordSystem(
	name='cylindrical2d',
	coords=['rho', 'phi'],
	inv_scale_factors=[_base.ConstFunc(1, input_dim=2), _base.CoordPow(input_dim=2, coord_index=0, power=-1, coord_name='rho')],
	to_cartesian=_polar_to_cart,
	from_cartesian=_cart_to_polar,
)

Polar3D = CoordSystem(
	name='spherical',
	coords=['r', 'theta', 'phi'],
	inv_scale_factors=[
		_base.ConstFunc(1, input_dim=3),
		_base.CoordPow(input_dim=3, coord_index=0, power=-1, coord_name='r'),
		_base.CoordPow(input_dim=3, coord_index=0, power=-1, coord_name='r') * _base.TrigCoord('csc', input_dim=3, coord_index=1, coord_name='theta'),
	],
	to_cartesian=_sph_to_cart,
	from_cartesian=_cart_to_sph,
)

_COORD_REGISTRY = {
	'cartesian2d':  Cartesian2D,
	'cartesian3d':  Cartesian3D,
	'cartesian':    Cartesian3D,
	'cylindrical':  Cylindrical3D,
	'cylindrical2d': Cylindrical2D,
	'polar':        Polar2D,
	'spherical':    Polar3D,
}
