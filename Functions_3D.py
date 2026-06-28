import numpy as np
from . import Functions_Base as _base
from . import CoordSystems as _coords
from .Functions_1D import ExpFunc, PowFunc, Bessel1D

_KNOWN_COORDS = frozenset({'x', 'y', 'z', 'rho', 'phi', 'r', 'theta'})


################
class Funcs3D(_base.FuncBase):

	def __init__(self, domain=lambda pos: True, output_dim=1):
		super().__init__(domain=domain, input_dim=3, output_dim=output_dim)

	def __setattr__(self, name, value):
		if name == 'coord_sys':
			raise AttributeError("coord_sys is a class-level constant and cannot be set on instances")
		super().__setattr__(name, value)

	def __call__(self, pos, ignore_domain=False): # TODO: This cannot take a combination of CoordPoint and ndarrays. Not a big problem, just an inconvenience.
		# CoordPoint input is converted to the function's native system before evaluation, so the
		# input may be in any system (and a batch may mix systems) — convert_to is a no-op when a
		# point is already in the target system.
		func_coord_sys = getattr(self, 'coord_sys', None)
		if isinstance(pos, _coords.CoordPoint):
			if func_coord_sys is not None:
				pos = pos.convert_to(func_coord_sys)
			pos = pos.pos
		elif isinstance(pos, list) and pos and isinstance(pos[0], _coords.CoordPoint):
			if func_coord_sys is not None:
				pos = [p.convert_to(func_coord_sys) for p in pos]
			pos = np.stack([p.pos for p in pos])
		return super().__call__(pos, ignore_domain)

	def derivative(self, coord):
		method = getattr(self, f'_deriv_{coord}', None)
		if method is not None:
			return method()
		if coord in _KNOWN_COORDS:
			raise NotImplementedError(f"{type(self).__name__} does not implement derivative for coord='{coord}'")
		raise ValueError(f"Unknown coordinate '{coord}'")

	def numerical_laplacian(self, pos, coord_sys, eps=1e-5):
		from .CoordSystems import _COORD_REGISTRY
		if isinstance(coord_sys, str):
			coord_sys = _COORD_REGISTRY[coord_sys]

		pos_arr, single_input = self.pos_to_arr(pos)
		n = len(coord_sys.coords)
		eps_list = [float(eps)] * n if isinstance(eps, (int, float)) else list(eps)

		inv_hs = coord_sys.inv_scale_factors
		steps = [np.eye(n)[i] * eps_list[i] for i in range(n)]

		def jacobian(pts):
			result = np.ones(len(pts))
			for ih in inv_hs:
				result = result / ih(pts, ignore_domain=True)
			return result

		f_center = self(pos_arr, ignore_domain=True)
		terms = []
		for i, ei in enumerate(steps):
			f_fwd = self(pos_arr + 2*ei, ignore_domain=True)
			f_bwd = self(pos_arr - 2*ei, ignore_domain=True)
			phi_plus  = jacobian(pos_arr + ei) * inv_hs[i](pos_arr + ei, ignore_domain=True)**2 * (f_fwd    - f_center) / (2 * eps_list[i])
			phi_minus = jacobian(pos_arr - ei) * inv_hs[i](pos_arr - ei, ignore_domain=True)**2 * (f_center - f_bwd   ) / (2 * eps_list[i])
			terms.append((phi_plus - phi_minus) / (2 * eps_list[i]))

		result = sum(terms) / jacobian(pos_arr)
		return result[0] if single_input else result



################
class Exp3D(_base._SeparableMixin, Funcs3D):

	coord_sys = _coords.Cartesian3D

	def __new__(cls, k_vec, ampl=1, domain=lambda pos: True):
		if ampl == 0:
			return _base.ZeroFunc(domain=domain, input_dim=3)
		return object.__new__(cls)

	def __init__(self, k_vec, ampl=1, domain=lambda pos: True):
		super().__init__(domain=domain)
		self.k_vec = np.array(k_vec)
		self.ampl = ampl
		self.parameters = {'k_vec': self.k_vec, 'ampl': ampl, 'domain': domain}

	def _factors(self):
		# Separable: e^(i·kx·x) · e^(i·ky·y) · e^(i·kz·z), one factor per [x, y, z].
		return [ExpFunc(k=1j * ki) for ki in self.k_vec]

	def __add__(self, other):
		if isinstance(other, Exp3D) and np.array_equal(self.k_vec, other.k_vec):
			return Exp3D(self.k_vec, ampl=self.ampl + other.ampl, domain=_base.combine_domains(self, other))
		return super().__add__(other)


################
# The function bess_fn(scale*rho) * e^(i*m_azim*phi) * e^(i*kz*z)
# bessel = [fn, order, scale] where fn in {'J','Y','I','K'} and the function is fn(order, scale*rho)
class Cylindrical(_base._SeparableMixin, Funcs3D):

	coord_sys = _coords.Cylindrical3D

	def __new__(cls, kz, m_azim, bessel, ampl=1, domain=lambda pos: True):
		if ampl == 0:
			return _base.ZeroFunc(domain=domain, input_dim=3)
		return object.__new__(cls)

	# domain function should take a 3-component vector input as [rho, phi, z]
	def __init__(self, kz, m_azim, bessel, ampl=1, domain=lambda pos: True):
		super().__init__(domain=domain)
		self.ampl = ampl
		self.kz = kz
		self.m_azim = m_azim
		self.bessel_name = bessel[0]
		self.order = bessel[1]
		self.scale = bessel[2] if len(bessel) == 3 else kz
		self.parameters = {'kz': kz, 'm_azim': m_azim,
						   'bessel': [self.bessel_name, self.order, self.scale],
						   'ampl': ampl, 'domain': domain}

	def _factors(self):
		# Separable: Bessel(scale·rho) · e^(i·m·phi) · e^(i·kz·z), one factor per [rho, phi, z]. _eval/derivative/sympy_output are derived from these by _SeparableMixin.
		return [Bessel1D(self.bessel_name, self.order, self.scale),
				ExpFunc(k=1j * self.m_azim),
				ExpFunc(k=1j * self.kz)]

	def _gradient_component(self, coord):
		if coord == 'phi':
			# Bessel recurrence B_n(x)/x = 1/(2n)·(B_{n-1}(x) ± B_{n+1}(x)) absorbs 1/ρ analytically, keeping result as a sum of Cylindrical terms.
			if self.m_azim == 0:
				return _base.ZeroFunc(domain=self.domain, input_dim=3)
			if self.order == 0:
				# recurrence needs 1/(2·order); let gradient() fall back to (1/ρ)·∂_φ = inv_h·derivative
				return NotImplemented
			if self.bessel_name in ['J','Y']:
				ampl_minus, ampl_plus = 1, 1
			elif self.bessel_name == 'I':
				ampl_minus, ampl_plus = 1, -1
			elif self.bessel_name == 'K':
				ampl_minus, ampl_plus = -1, 1
			else:
				raise ValueError(f"Unknown Bessel function: {self.bessel_name}")
			ampl = 1j * self.m_azim * self.scale / (2 * self.order)
			return (ampl*ampl_plus)*self.increased_order() + (ampl*ampl_minus)*self.reduced_order()
		return NotImplemented

	def __add__(self, other):
		if (isinstance(other, Cylindrical)
				and self.kz == other.kz and self.m_azim == other.m_azim
				and self.bessel_name == other.bessel_name
				and self.order == other.order and self.scale == other.scale):
			return Cylindrical(self.kz, self.m_azim,
							   [self.bessel_name, self.order, self.scale],
							   ampl=self.ampl + other.ampl, domain=_base.combine_domains(self, other))
		return super().__add__(other)

	def reduced_order(self):
		return self.clone(bessel=[self.bessel_name, self.order - 1, self.scale])

	def increased_order(self):
		return self.clone(bessel=[self.bessel_name, self.order + 1, self.scale])


################
# The function ampl * rho^power * e^(i*m_azim*phi) * e^(i*kz*z)
class PowerCylindrical(_base._SeparableMixin, Funcs3D):

	coord_sys = _coords.Cylindrical3D

	def __new__(cls, kz, m_azim, power=0, ampl=1, domain=lambda pos: True):
		if ampl == 0:
			return _base.ZeroFunc(domain=domain, input_dim=3)
		return object.__new__(cls)

	def __init__(self, kz, m_azim, power=0, ampl=1, domain=lambda pos: True):
		super().__init__(domain=domain)
		self.kz = kz
		self.m_azim = m_azim
		self.power = power
		self.ampl = ampl
		self.parameters = {'kz': kz, 'm_azim': m_azim, 'power': power, 'ampl': ampl, 'domain': domain}

	def _factors(self):
		# Separable: rho^power · e^(i·m·phi) · e^(i·kz·z), one factor per [rho, phi, z].
		return [PowFunc(self.power), ExpFunc(k=1j * self.m_azim), ExpFunc(k=1j * self.kz)]

	def _gradient_component(self, coord):
		if coord == 'phi':
			# (1/rho)·∂_φ f absorbs the scale factor analytically: drops the power by one.
			if self.m_azim == 0:
				return _base.ZeroFunc(domain=self.domain, input_dim=3)
			return self.clone(power=self.power - 1, ampl=1j * self.m_azim * self.ampl)
		return NotImplemented

	def __add__(self, other):
		if (isinstance(other, PowerCylindrical)
				and self.kz == other.kz and self.m_azim == other.m_azim
				and self.power == other.power):
			return PowerCylindrical(self.kz, self.m_azim, power=self.power,
									ampl=self.ampl + other.ampl, domain=_base.combine_domains(self, other))
		return super().__add__(other)
