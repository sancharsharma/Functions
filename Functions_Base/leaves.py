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

	def simplify(self, deep=False):
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
		from ..Functions_1D import PowFunc   # deferred: breaks the Functions_Base ↔ Functions_1D import cycle
		return PowFunc(power=1, ampl=self.const)

	def sympy_output(self):
		if self.output_dim == 1:
			return sym.sympify(self.const)
		return sym.Matrix(self.const.tolist())


######### Lifting a 1-D function into n dimensions
class Embed1D(FuncBase):
	"""Lift a 1-D function into an ``input_dim``-dimensional one by evaluating it on a single coordinate: ``Embed1D(f, n, i)(pos) = f(pos[i])``. The wrapped ``func`` is always a 1-D function (input_dim=1, e.g. ``PowFunc``, ``Sin``, ``Cos``); this class supplies the column extraction, the partial-derivative dispatch on ``coord_name`` (the embedded variable; every other coordinate differentiates to zero), and the SymPy variable relabelling. Its internals (``_func``, ``_coord_index``, ``_coord_name``) are implementation detail. As the embedding is linear and identity-preserving, a zero/constant inner func lifts to ``ZeroFunc``/``ConstFunc`` (collapsed in ``__new__``). Coordinate scale factors in CoordSystem are built from it."""

	def __new__(cls, func, input_dim, coord_index, coord_name=None, domain=lambda _: True):
		if isinstance(func, ZeroFunc):
			return ZeroFunc(domain=domain, input_dim=input_dim)
		if isinstance(func, ConstFunc):
			return ConstFunc(func.const, domain=domain, input_dim=input_dim)
		return object.__new__(cls)

	def __init__(self, func, input_dim, coord_index, coord_name=None, domain=lambda _: True): # TODO: Should domain be given or inferred from func? Or should we combine the two domains?
		super().__init__(domain=domain, input_dim=input_dim, output_dim=1)
		self._func = func
		self._coord_index = coord_index
		self._coord_name = coord_name if coord_name is not None else f'x{coord_index}'
		self.parameters = {'func': func, 'input_dim': input_dim, 'coord_index': coord_index, 'coord_name': self._coord_name, 'domain': domain}

	def _eval(self, pos_arr):
		col = pos_arr if self.input_dim == 1 else pos_arr[:, self._coord_index]
		return self._func._eval(col)

	def derivative(self, coord, *args, **kwargs):
		if coord == self._coord_name:
			return self.clone(func=self._func.derivative())
		return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=1)

	def reciprocal(self):
		return self.clone(func=self._func.reciprocal())

	def sympy_output(self):
		return self._func.sympy_output().subs(sym.Symbol('x', real=True), sym.Symbol(self._coord_name, real=True))


# ConstFunc.__mul__ builds a ProdOfFuncs by name; deferred to module-bottom to break the import
# cycle (combinators subclass FuncBase and reference ConstFunc/ZeroFunc).
from .combinators import ProdOfFuncs
