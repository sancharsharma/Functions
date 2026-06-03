import numpy as np
from abc import abstractmethod
import scipy.special as spec
import sympy as sym
from . import Functions_Base as _base
from .CoordSystems import Cylindrical3D, Cartesian3D, CoordPoint
import sympy.functions.special.bessel as bessel

_KNOWN_COORDS = frozenset({'x', 'y', 'z', 'rho', 'phi', 'r', 'theta'})


################
class Funcs3D(_base.FuncBase):

	def __init__(self, domain=lambda pos: True, output_dim=1):
		super().__init__(domain=domain, input_dim=3, output_dim=output_dim)

	def __setattr__(self, name, value):
		if name == 'coord_sys':
			raise AttributeError("coord_sys is a class-level constant and cannot be set on instances")
		super().__setattr__(name, value)

	# TODO: A list of coordinate points will not work here.
	def __call__(self, pos, ignore_domain=False):
		if isinstance(pos, CoordPoint):
			func_coord_sys = getattr(self, 'coord_sys', None)
			if func_coord_sys is not None and pos.coord_sys is not func_coord_sys:
				pos = pos.convert_to(func_coord_sys)
			pos = pos.pos
		return super().__call__(pos, ignore_domain)

	def derivative(self, coord):
		method = getattr(self, f'_deriv_{coord}', None)
		if method is not None:
			return method()
		if coord in _KNOWN_COORDS:
			raise NotImplementedError(f"{type(self).__name__} does not implement derivative for coord='{coord}'")
		raise ValueError(f"Unknown coordinate '{coord}'")

	@abstractmethod
	def sympy_output(self):
		pass

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
class Exp3D(Funcs3D):

	coord_sys = Cartesian3D

	def __new__(cls, k_vec, ampl=1, domain=lambda pos: True):
		if ampl == 0:
			return _base.ZeroFunc(domain=domain, input_dim=3)
		return object.__new__(cls)

	def __init__(self, k_vec, ampl=1, domain=lambda pos: True):
		super().__init__(domain=domain)
		self.k_vec = np.array(k_vec)
		self.ampl = ampl

	def _eval(self, pos_arr):
		phase = np.dot(pos_arr, self.k_vec)
		return self.ampl * np.exp(1j * phase)

	def __add__(self, other):
		if isinstance(other, Exp3D) and np.array_equal(self.k_vec, other.k_vec):
			domain = lambda pos: self.domain(pos) and other.domain(pos)
			return Exp3D(self.k_vec, ampl=self.ampl + other.ampl, domain=domain)
		return super().__add__(other)

	def copy(self):
		return Exp3D(self.k_vec.copy(), ampl=self.ampl, domain=self.domain)

	def _deriv_x(self):
		return Exp3D(self.k_vec, ampl=1j*self.k_vec[0]*self.ampl, domain=self.domain)

	def _deriv_y(self):
		return Exp3D(self.k_vec, ampl=1j*self.k_vec[1]*self.ampl, domain=self.domain)

	def _deriv_z(self):
		return Exp3D(self.k_vec, ampl=1j*self.k_vec[2]*self.ampl, domain=self.domain)

	def sympy_output(self):
		x, y, z = sym.symbols('x y z', real=True)
		phase = sum((ki*ci for ki, ci in zip(self.k_vec, [x, y, z])), sym.S.Zero)
		return self.ampl * sym.exp(sym.I * phase)


################
# The function bess_fn(scale*rho) * e^(i*m_azim*phi) * e^(i*kz*z)
# bessel = [fn, order, scale] where fn in {'J','Y','I','K'} and the function is fn(order, scale*rho)
class Cylindrical(Funcs3D):

	coord_sys = Cylindrical3D

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

		match bessel[0]:
			case 'J':
				self.bess_fn = spec.jv
			case 'Y':
				self.bess_fn = spec.yn if isinstance(self.order, int) else spec.yv
			case 'I':
				self.bess_fn = spec.iv
			case 'K':
				self.bess_fn = spec.kn if isinstance(self.order, int) else spec.kv

	def _eval(self, pos_arr):
		bess_out = self.bess_fn(self.order, self.scale*pos_arr[:,0])
		azim_out = np.exp(1j*self.m_azim*pos_arr[:,1])
		z_out    = np.exp(1j*self.kz*pos_arr[:,2])
		return self.ampl*bess_out*azim_out*z_out

	def _deriv_rho(self):
		if self.bessel_name in ['J','Y']:
			ampl_minus, ampl_plus = 1, -1
		elif self.bessel_name == 'I':
			ampl_minus, ampl_plus = 1, 1
		elif self.bessel_name == 'K':
			ampl_minus, ampl_plus = -1, -1
		else:
			raise ValueError(f"Unknown Bessel function: {self.bessel_name}")
		return (ampl_plus*self.scale/2)*self.increased_order() + (ampl_minus*self.scale/2)*self.reduced_order()

	def _deriv_phi(self):
		if self.m_azim == 0:
			return _base.ZeroFunc(domain=self.domain, input_dim=3)
		return 1j * self.m_azim * self

	def _gradient_component(self, coord):
		if self.coord_sys.name == 'cylindrical' and coord == 'phi':
			# Bessel recurrence B_n(x)/x = 1/(2n)·(B_{n-1}(x) ± B_{n+1}(x)) absorbs 1/ρ analytically, keeping result as a sum of Cylindrical terms.
			if self.m_azim == 0 or self.order == 0:
				return _base.ZeroFunc(domain=self.domain, input_dim=3)
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

	def _deriv_z(self):
		return 1j*self.kz*self

	def __add__(self, other):
		if (isinstance(other, Cylindrical)
				and self.kz == other.kz and self.m_azim == other.m_azim
				and self.bessel_name == other.bessel_name
				and self.order == other.order and self.scale == other.scale):
			domain = lambda pos: self.domain(pos) and other.domain(pos)
			return Cylindrical(self.kz, self.m_azim,
							   [self.bessel_name, self.order, self.scale],
							   ampl=self.ampl + other.ampl, domain=domain)
		return super().__add__(other)

	def __mul__(self, other):
		if np.isscalar(other):
			return Cylindrical(self.kz, self.m_azim, [self.bessel_name, self.order, self.scale], ampl=other*self.ampl, domain=self.domain)
		return super().__mul__(other)

	def __rmul__(self, other):
		return self.__mul__(other)

	def reduced_order(self):
		return Cylindrical(self.kz, self.m_azim, [self.bessel_name, self.order-1, self.scale], ampl=self.ampl, domain=self.domain)

	def increased_order(self):
		return Cylindrical(self.kz, self.m_azim, [self.bessel_name, self.order+1, self.scale], ampl=self.ampl, domain=self.domain)

	def copy(self):
		return Cylindrical(self.kz, self.m_azim, [self.bessel_name, self.order, self.scale], ampl=self.ampl, domain=self.domain)

	def sympy_output(self):
		rho, phi, z = sym.symbols('rho phi z', real=True)
		bessel_map = {'J': bessel.besselj, 'Y': bessel.bessely, 'I': bessel.besseli, 'K': bessel.besselk}
		fun = bessel_map[self.bessel_name]
		return self.ampl * fun(self.order, self.scale*rho) * sym.exp(sym.I*self.kz*z) * sym.exp(sym.I*self.m_azim*phi)


################
# The function ampl * rho^power * e^(i*m_azim*phi) * e^(i*kz*z)
class PowerCylindrical(Funcs3D):

	coord_sys = Cylindrical3D

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

	def _eval(self, pos_arr):
		rho_out  = pos_arr[:,0] ** self.power
		azim_out = np.exp(1j * self.m_azim * pos_arr[:,1])
		z_out    = np.exp(1j * self.kz * pos_arr[:,2])
		return self.ampl * rho_out * azim_out * z_out

	def _deriv_rho(self):
		if self.power == 0:
			return _base.ZeroFunc(domain=self.domain, input_dim=3)
		return PowerCylindrical(self.kz, self.m_azim, power=self.power-1, ampl=self.power*self.ampl, domain=self.domain)

	def _deriv_z(self):
		return PowerCylindrical(self.kz, self.m_azim, power=self.power, ampl=1j*self.kz*self.ampl, domain=self.domain)

	def _deriv_phi(self):
		if self.m_azim == 0:
			return _base.ZeroFunc(domain=self.domain, input_dim=3)
		return 1j * self.m_azim * self

	def _gradient_component(self, coord):
		if coord == 'phi':
			if self.m_azim == 0:
				return _base.ZeroFunc(domain=self.domain, input_dim=3)
			return PowerCylindrical(self.kz, self.m_azim, power=self.power-1, ampl=1j*self.m_azim*self.ampl, domain=self.domain)
		return NotImplemented

	def __add__(self, other):
		if (isinstance(other, PowerCylindrical)
				and self.kz == other.kz and self.m_azim == other.m_azim
				and self.power == other.power):
			domain = lambda pos: self.domain(pos) and other.domain(pos)
			return PowerCylindrical(self.kz, self.m_azim, power=self.power,
									ampl=self.ampl + other.ampl, domain=domain)
		return super().__add__(other)

	def __mul__(self, other):
		if np.isscalar(other):
			return PowerCylindrical(self.kz, self.m_azim, power=self.power, ampl=other*self.ampl, domain=self.domain)
		return super().__mul__(other)

	def __rmul__(self, other):
		return self.__mul__(other)

	def copy(self):
		return PowerCylindrical(self.kz, self.m_azim, power=self.power, ampl=self.ampl, domain=self.domain)

	def sympy_output(self):
		rho, phi, z = sym.symbols('rho phi z', real=True)
		return self.ampl * rho**self.power * sym.exp(sym.I*self.m_azim*phi) * sym.exp(sym.I*self.kz*z)

