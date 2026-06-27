import numpy as np
import sympy as sym

from .func_base import FuncBase, _check_dim_compat, combine_domains, as_scalar, _SENTINEL

######## Some basic functions

class ZeroFunc(FuncBase):

	def __new__(cls, domain=lambda _: True, *, input_dim, output_dim=1):
		return object.__new__(cls)

	def __init__(self, domain=lambda _: True, *, input_dim, output_dim=1):
		super().__init__(domain=domain, input_dim=input_dim, output_dim=output_dim)
		self.parameters = {'domain': domain, 'input_dim': input_dim, 'output_dim': output_dim}

	def _eval(self, pos_arr):
		return np.zeros((len(pos_arr), self.output_dim)) if self.output_dim > 1 else np.zeros(len(pos_arr))

	def __add__(self, other):
		if isinstance(other, FuncBase):
			_check_dim_compat([self, other], 'sum')
			return other
		if np.isscalar(other) and self.output_dim == 1:
			return ConstFunc(other, input_dim=self.input_dim)
		other_arr = np.asarray(other)
		if other_arr.ndim == 1 and len(other_arr) == self.output_dim:
			return ConstFunc(other_arr, input_dim=self.input_dim)
		return NotImplemented

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		if np.isscalar(other):
			return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)
		if isinstance(other, FuncBase):
			input_dim, output_dim = _check_dim_compat([self, other], 'mul')
			return ZeroFunc(domain=combine_domains(self, other), input_dim=input_dim, output_dim=output_dim)
		return NotImplemented

	def derivative(self, *_args, **_kwargs):
		return self

	def antiderivative(self):
		return self.copy()

	def sympy_output(self):
		if self.output_dim > 1:
			return sym.zeros(self.output_dim, 1)
		return sym.Integer(0)

	def simplify(self, _deep=False):
		return self


class ConstFunc(FuncBase):

	def __new__(cls, const, domain=lambda _: True, *, input_dim):
		arr = np.asarray(const)
		if np.all(arr == 0):
			inferred = int(arr.size) if arr.ndim > 0 else 1
			return ZeroFunc(domain=domain, input_dim=input_dim, output_dim=inferred)
		return object.__new__(cls)

	def __init__(self, const, domain=lambda _: True, *, input_dim):
		arr = np.asarray(const)
		if arr.ndim == 0:
			self.const = arr.item()
			inferred = 1
		else:
			self.const = arr.copy()
			inferred = len(arr)
		super().__init__(domain=domain, input_dim=input_dim, output_dim=inferred)
		self.parameters = {'const': self.const, 'domain': domain, 'input_dim': input_dim}

	def _eval(self, pos_arr):
		if self.output_dim == 1:
			return np.full(len(pos_arr), self.const)
		return np.tile(self.const, (len(pos_arr), 1))

	def __add__(self, other):
		if isinstance(other, ConstFunc):
			input_dim, _ = _check_dim_compat([self, other], 'sum')
			return ConstFunc(self.const + other.const, domain=combine_domains(self, other), input_dim=input_dim)
		if np.isscalar(other) and self.output_dim == 1:
			return ConstFunc(self.const + other, input_dim=self.input_dim)
		return super().__add__(other)

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		val = as_scalar(other)
		if val is not None:
			return ConstFunc(val * self.const, domain=self.domain, input_dim=self.input_dim)
		if isinstance(other, ConstFunc):
			input_dim, _ = _check_dim_compat([self, other], 'mul')
			return ConstFunc(self.const * other.const, domain=combine_domains(self, other), input_dim=input_dim)
		if isinstance(other, FuncBase):
			_check_dim_compat([self, other], 'mul')
			result = other.__mul__(self)
			if result is not NotImplemented:
				return result
			return ProdOfFuncs([self, other], _key=_SENTINEL)
		return NotImplemented

	def reciprocal(self):
		return ConstFunc(1 / self.const, domain=self.domain, input_dim=self.input_dim)

	def derivative(self, *args, **kwargs):
		return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)

	def antiderivative(self):
		if self.input_dim != 1 or self.output_dim != 1:
			raise NotImplementedError(
				"antiderivative is only defined for 1D scalar functions (input_dim=output_dim=1)"
			)
		return self.const * CoordPow(1, 0, 1, coord_name='x')

	def sympy_output(self):
		if self.output_dim == 1:
			return sym.sympify(self.const)
		return sym.Matrix(self.const.tolist())


######### Coordinate transformation functions
class CoordPow(FuncBase):
	"""pos[coord_index]^power for any input_dim — used as coordinate scale factors in CoordSystem."""

	def __new__(cls, input_dim, coord_index, power=1, coord_name=None, domain=lambda _: True):
		if power == 0:
			return ConstFunc(1, domain=domain, input_dim=input_dim)
		return object.__new__(cls)

	def __init__(self, input_dim, coord_index, power=1, coord_name=None, domain=lambda _: True):
		super().__init__(domain=domain, input_dim=input_dim, output_dim=1)
		self.coord_index = coord_index
		self.power = power
		self.coord_name = coord_name if coord_name is not None else f'x{coord_index}'
		self.parameters = {'input_dim': input_dim, 'coord_index': coord_index, 'power': power,
						   'coord_name': self.coord_name, 'domain': domain}

	def _eval(self, pos_arr):
		col = pos_arr if self.input_dim == 1 else pos_arr[:, self.coord_index]
		return col ** self.power

	def derivative(self, coord, *args, **kwargs):
		if coord == self.coord_name:
			return self.power * self.clone(power=self.power - 1)
		return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=1)

	def reciprocal(self):
		return self.clone(power=-self.power)

	def sympy_output(self):
		return sym.Symbol(self.coord_name, real=True) ** self.power


class TrigCoord(FuncBase):
	"""sin or cos of one coordinate of a multi-dimensional input. (csc/sec were removed; take the
	reciprocal of a TrigCoord to get 1/sin or 1/cos as a RatioOfFuncs.)"""

	_funcs = ('sin', 'cos')

	def __init__(self, func, input_dim, coord_index, coord_name=None, domain=lambda _: True):
		if func not in self._funcs:
			raise ValueError(f"TrigCoord func must be one of {list(self._funcs)}, got '{func}'")
		super().__init__(domain=domain, input_dim=input_dim, output_dim=1)
		self.func = func
		self.coord_index = coord_index
		self.coord_name = coord_name if coord_name is not None else f'x{coord_index}'
		self.parameters = {'func': func, 'input_dim': input_dim, 'coord_index': coord_index,
						   'coord_name': self.coord_name, 'domain': domain}

	def _eval(self, pos_arr):
		col = pos_arr[:, self.coord_index]
		return np.sin(col) if self.func == 'sin' else np.cos(col)

	def derivative(self, coord, *args, **kwargs):
		if coord != self.coord_name:
			return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=1)
		if self.func == 'sin':
			return self.clone(func='cos')
		return ConstFunc(-1, input_dim=self.input_dim) * self.clone(func='sin')

	def sympy_output(self):
		s = sym.Symbol(self.coord_name, real=True)
		return sym.sin(s) if self.func == 'sin' else sym.cos(s)


# ConstFunc.__mul__ builds a ProdOfFuncs by name; deferred to module-bottom to break the import
# cycle (combinators subclass FuncBase and reference ConstFunc/ZeroFunc).
from .combinators import ProdOfFuncs
