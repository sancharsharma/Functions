import numpy as np
from abc import ABC, abstractmethod
import sympy as sym


class FuncBase(ABC):

	def __init__(self, domain=lambda _: True, input_dim=None, output_dim=None):
		self.domain = domain
		self.input_dim = input_dim
		self.output_dim = output_dim

	def pos_to_arr(self, pos):
		d = self.input_dim
		if d == 0: # Discrete space
			arr = np.asarray(pos)
			if not np.issubdtype(arr.dtype, np.integer):
				raise ValueError(f"input_dim=0 (discrete) requires integer input, got dtype {arr.dtype}")
			if arr.ndim == 0:  return arr.reshape(1), True
			if arr.ndim == 1:  return arr, False
			raise ValueError("Discrete function expects a scalar integer or 1D integer array")
		pos_arr = np.asarray(pos)
		if np.issubdtype(pos_arr.dtype, np.complexfloating):
			raise ValueError(f"Inputs must be real-valued, got dtype {pos_arr.dtype}. Complex inputs are not supported.")
		pos_arr = pos_arr.astype(float)
		if d == 1:
			if pos_arr.ndim == 0:   return pos_arr.reshape(1), True
			if pos_arr.ndim == 1:   return pos_arr, False
			raise ValueError("1D function expects a scalar or 1D array")
		if d is not None:
			if pos_arr.shape == (d,):                          return pos_arr.reshape(1, d), True
			if pos_arr.ndim == 2 and pos_arr.shape[1] == d:   return pos_arr, False
			raise ValueError(f"{d}D function expects a {d}-vector or an (N, {d}) array")
		raise ValueError(f"input_dim must be set, got {self.input_dim!r}")


	# Any class that subclasses FuncBase must implement all the methods decorated with @abstractmethod, and must call super().__init__() in its __init__ method. Its adder and multiplication methods must return NotImplemented if the other object is not of a type that the method explicitly handles. The copy method must return a new instance of the same class with the same parameters, and the sympy_output method must return a sympy expression representing the function using self.sympy_var as the free symbol for 1D functions. The derivative method must return a new FuncBase object representing the derivative of the function.
	@abstractmethod
	def _eval(self, pos_arr): ...

	@abstractmethod
	def derivative(self, *args, **kwargs):
		"""Return a new FuncBase representing the derivative.

		Calling convention depends on the subclass family:
		  - 1D  (Funcs1D subclasses): no arguments — always w.r.t. the single variable.
		  - 3D  (Funcs3D subclasses): derivative(coord: str) — coordinate name, e.g. 'rho', 'phi', 'z'.
		  - Discrete (FuncsDiscrete): derivative(direction='forward') — 'forward', 'backward', or 'central'.

		Combinators (SumOfFuncs, ProdOfFuncs, etc.) forward *args/**kwargs unchanged.
		"""
		...

	@abstractmethod
	def sympy_output(self): ...

	@abstractmethod
	def copy(self): ...

	def __repr__(self):
		return (f"<{type(self).__name__}>"
				" (call .sympy_output() for a symbolic representation — may be slow for large expressions)")

	def numerical_derivative(self, pos, coord_index=0, eps=1e-5):
		if self.input_dim == 0:
			raise TypeError(
				"numerical_derivative is not defined for discrete (input_dim=0) functions; "
				"use .derivative(direction='forward'/'backward') instead."
			)
		pos_arr, single_input = self.pos_to_arr(pos)
		if self.input_dim == 1:
			eps_step = eps
		else:
			eps_step = np.zeros(self.input_dim)
			eps_step[coord_index] = eps
		result = (self(pos_arr + eps_step, ignore_domain=True) - self(pos_arr - eps_step, ignore_domain=True)) / (2 * eps)
		return result[0] if single_input else result

	def reciprocal(self):
		if self.output_dim is not None and self.output_dim > 1:
			numer = ConstFunc(np.ones(self.output_dim), input_dim=self.input_dim)
		else:
			numer = ConstFunc(1, input_dim=self.input_dim, output_dim=1)
		return RatioOfFuncs(numer, self, _key=_SENTINEL)

	def gradient(self, coord_sys=None):
		if coord_sys is None:
			coord_sys = getattr(self, 'coord_sys', None)
		if coord_sys is None:
			raise ValueError(f"gradient() requires coord_sys for {type(self).__name__}, which has no natural coordinate system")
		return coord_sys.gradient(self)

	def laplacian(self, coord_sys=None):
		if coord_sys is None:
			coord_sys = getattr(self, 'coord_sys', None)
		if coord_sys is None:
			raise ValueError(f"laplacian() requires coord_sys for {type(self).__name__}, which has no natural coordinate system")
		return coord_sys.laplacian(self)

	def numerical_integrate(self, a, b):
		from scipy.integrate import quad
		if self.input_dim is not None and self.input_dim != 1:
			raise NotImplementedError(
				f"numerical_integrate() is not supported for functions with input_dim={self.input_dim}; "
				"only 1D scalar functions support numerical integration"
			)
		if self.output_dim is not None and self.output_dim != 1:
			raise NotImplementedError(
				f"numerical_integrate() is not supported for vector-valued functions (output_dim={self.output_dim})"
			)
		result, _ = quad(lambda x: float(self(x, ignore_domain=True)), a, b)
		return result

	def integrate(self, a=None, b=None):
		if a is None and b is None:
			raise NotImplementedError(
				f"{type(self).__name__} does not implement indefinite integration; "
				"call integrate(a, b) for a definite integral"
			)
		if a is not None and b is not None:
			try:
				F = self.integrate()
				return F(b, ignore_domain=True) - F(a, ignore_domain=True)
			except NotImplementedError:
				return self.numerical_integrate(a, b)
		raise ValueError(
			"integrate() takes either no positional arguments (indefinite) "
			"or exactly two (definite interval [a, b])"
		)

	def __call__(self, pos, ignore_domain=False):
		pos_arr, single_input = self.pos_to_arr(pos)
		if not ignore_domain:
			self._check_domain(pos_arr)
		result = self._eval(pos_arr)
		return result[0] if single_input else result

	def _check_domain(self, pos_arr, output='error'):
		"""Domain check on an already-normalised pos_arr (N, d) or (N,) array."""
		try:
			bad_points = [p for p in pos_arr if not self.domain(p)]
		except Exception as e:
			raise ValueError(e)
		if output == 'points':
			return bad_points
		elif output == 'binary':
			return bad_points == []
		elif output == 'error':
			if bad_points:
				raise ValueError("There are points outside domain. Call with ignore_domain=True to bypass, or use check_domain(pos, output='points') to inspect.")

	def check_domain(self, pos, output='error'):
		"""Public domain check; accepts raw user input (scalar, array, etc.)."""
		pos_arr, _ = self.pos_to_arr(pos)
		return self._check_domain(pos_arr, output)

	def __neg__(self):
		return (-1) * self

	def __sub__(self, other):
		return self + (-1) * other

	def __add__(self, other):
		if isinstance(other, ZeroFunc):
			return self
		if np.isscalar(other):
			return SumOfFuncs([self, ConstFunc(other, input_dim=self.input_dim, output_dim=1)], _key=_SENTINEL)
		if isinstance(other, FuncBase):
			return SumOfFuncs([self, other], _key=_SENTINEL)
		return NotImplemented

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		if isinstance(other, ZeroFunc):
			od = next((d for d in (other.output_dim, self.output_dim) if d is not None), 1)
			return ZeroFunc(domain=lambda pos: self.domain(pos) and other.domain(pos), input_dim=self.input_dim, output_dim=od)
		if isinstance(other, FuncBase):
			return ProdOfFuncs([self, other], _key=_SENTINEL)
		if np.isscalar(other):
			return ProdOfFuncs([ConstFunc(other, input_dim=self.input_dim, output_dim=1), self], _key=_SENTINEL)
		arr = np.asarray(other)
		if arr.ndim == 1:
			if self.output_dim is not None and len(arr) != self.output_dim:
				raise ValueError(f"Cannot multiply output_dim={self.output_dim} function by length-{len(arr)} array")
			return ProdOfFuncs([ConstFunc(arr, input_dim=self.input_dim), self], _key=_SENTINEL)
		return NotImplemented

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		if np.isscalar(other):
			return self * (1.0 / other)
		if isinstance(other, FuncBase):
			return RatioOfFuncs(self, other, _key=_SENTINEL)
		return NotImplemented

	def __rtruediv__(self, other):
		if np.isscalar(other):
			return RatioOfFuncs(ConstFunc(other, input_dim=self.input_dim, output_dim=1), self, _key=_SENTINEL)
		return NotImplemented

	def __eq__(self, other):
		if not isinstance(other, FuncBase):
			return NotImplemented
		try:
			diff = self - other
			if hasattr(diff, 'simplify'):
				diff = diff.simplify()
			return isinstance(diff, ZeroFunc)
		except Exception:
			return NotImplemented

	__hash__ = object.__hash__

#    Sometimes this is too slow, so removed it
#    def __repr__(self):
#        return str(self.sympy_output())


def _check_dim_compat(funcs, mode):
	in_dims = {f.input_dim for f in funcs if f.input_dim is not None}
	if len(in_dims) > 1:
		raise ValueError(f"Cannot combine functions with different input_dim: {in_dims}")
	input_dim = in_dims.pop() if in_dims else None

	out_dims = [f.output_dim for f in funcs]
	if mode == 'sum':
		known = {d for d in out_dims if d is not None}
		if len(known) > 1:
			raise ValueError(f"incompatible output_dim for addition: {known}")
		output_dim = known.pop() if known else None
	else:  # 'mul'
		known = {d for d in out_dims if d is not None and d != 1}
		if len(known) > 1:
			raise ValueError(f"incompatible output_dim for multiplication: {known}")
		output_dim = known.pop() if known else (1 if 1 in out_dims else None)

	return input_dim, output_dim

######## Some basic functions

class ZeroFunc(FuncBase):

	def __new__(cls, domain=lambda _: True, input_dim=None, output_dim=1):
		if output_dim is None:
			raise ValueError(
				"ZeroFunc requires an explicit output_dim; None is ambiguous. "
				"Use output_dim=1 for a scalar zero, or output_dim=k for a vector zero of length k."
			)
		return object.__new__(cls)

	def __init__(self, domain=lambda _: True, input_dim=None, output_dim=1):
		super().__init__(domain=domain, input_dim=input_dim, output_dim=output_dim)

	def _eval(self, pos_arr):
		return np.zeros((len(pos_arr), self.output_dim)) if self.output_dim > 1 else np.zeros(len(pos_arr))

	def __add__(self, other):
		if isinstance(other, FuncBase):
			_check_dim_compat([self, other], 'sum')
			return other
		if np.isscalar(other) and self.output_dim == 1:
			return ConstFunc(other, input_dim=self.input_dim, output_dim=1)
		other_arr = np.asarray(other)
		if other_arr.ndim == 1 and len(other_arr) == self.output_dim:
			return ConstFunc(other_arr, input_dim=self.input_dim, output_dim=self.output_dim)
		return NotImplemented

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		if np.isscalar(other):
			return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)
		if isinstance(other, FuncBase):
			input_dim, output_dim = _check_dim_compat([self, other], 'mul')
			if output_dim is None:
				output_dim = 1
			return ZeroFunc(domain=lambda pos: self.domain(pos) and other.domain(pos), input_dim=input_dim, output_dim=output_dim)
		return NotImplemented

	def __rmul__(self, other):
		return self.__mul__(other)

	def derivative(self, *_args, **_kwargs):
		return self

	def integrate(self, a=None, b=None):
		if a is None and b is None:
			return self.copy()
		return super().integrate(a, b)

	def sympy_output(self):
		if self.output_dim > 1:
			return sym.zeros(self.output_dim, 1)
		return sym.Integer(0)

	def simplify(self, _deep=False):
		return self

	def copy(self):
		return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)


class ConstFunc(FuncBase):

	def __new__(cls, const, domain=lambda _: True, input_dim=None, output_dim=None):
		arr = np.asarray(const)
		if np.all(arr == 0):
			inferred = int(arr.size) if arr.ndim > 0 else 1
			return ZeroFunc(domain=domain, input_dim=input_dim, output_dim=output_dim if output_dim is not None else inferred)
		return object.__new__(cls)

	def __init__(self, const, domain=lambda _: True, input_dim=None, output_dim=None):
		arr = np.asarray(const)
		if arr.ndim == 0:
			self.const = arr.item()
			inferred = 1
		else:
			self.const = arr.copy()
			inferred = len(arr)
		if output_dim is not None and output_dim != inferred:
			raise ValueError(f"output_dim={output_dim} does not match inferred {inferred} from const shape")
		super().__init__(domain=domain, input_dim=input_dim, output_dim=inferred)

	def _eval(self, pos_arr):
		if self.output_dim == 1:
			return np.full(len(pos_arr), self.const)
		return np.tile(self.const, (len(pos_arr), 1))

	def __add__(self, other):
		if isinstance(other, ConstFunc):
			input_dim, _ = _check_dim_compat([self, other], 'sum')
			return ConstFunc(self.const + other.const, domain=lambda pos: self.domain(pos) and other.domain(pos), input_dim=input_dim, output_dim=self.output_dim)
		if np.isscalar(other) and self.output_dim == 1:
			return ConstFunc(self.const + other, input_dim=self.input_dim, output_dim=1)
		return super().__add__(other)

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		if np.isscalar(other):
			return ConstFunc(other * self.const, domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)
		if isinstance(other, ConstFunc):
			input_dim, output_dim = _check_dim_compat([self, other], 'mul')
			return ConstFunc(self.const * other.const, domain=lambda pos: self.domain(pos) and other.domain(pos), input_dim=input_dim, output_dim=output_dim)
		if isinstance(other, FuncBase):
			_check_dim_compat([self, other], 'mul')
			result = other.__mul__(self)
			if result is not NotImplemented:
				return result
			return ProdOfFuncs([self, other], _key=_SENTINEL)
		return NotImplemented

	def __rmul__(self, other):
		return self.__mul__(other)

	def reciprocal(self):
		return ConstFunc(1 / self.const, domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)

	def derivative(self, *args, **kwargs):
		return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)

	def integrate(self, a=None, b=None):
		if a is None and b is None:
			if self.output_dim != 1:
				raise NotImplementedError(
					f"indefinite integration of vector ConstFunc (output_dim={self.output_dim}) is not supported"
				)
			return self.const * CoordPow(self.input_dim or 1, 0, 1, coord_name='x')
		if self.output_dim != 1:
			return super().integrate(a, b)
		return self.const * (b - a)

	def sympy_output(self):
		if self.output_dim == 1:
			return sym.sympify(self.const)
		return sym.Matrix(self.const.tolist())

	def copy(self):
		return ConstFunc(self.const, domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)

######### Different ways to combine functions

_SENTINEL = object() # Used to control internal constructor behavior of these functions

class SumOfFuncs(FuncBase):
	# gradient()/laplacian() without an explicit coord_sys will raise ValueError here since SumOfFuncs
	# has no coord_sys attribute. If all terms share the same coord_sys, pass it explicitly. Terms from
	# different coordinate systems will also fail inside CoordSystem.gradient at the derivative() call.

	def __new__(cls, funcs, _key=None):
		if _key is not _SENTINEL:
			raise TypeError("SumOfFuncs is internal — combine functions with + instead.")
		if len(funcs) == 0:
			return ZeroFunc()
		if len(funcs) == 1:
			return funcs[0]
		return object.__new__(cls)

	def __init__(self, funcs, _key=None):
		input_dim, output_dim = _check_dim_compat(funcs, 'sum')
		domain = lambda pos: all(f.domain(pos) for f in funcs)
		super().__init__(domain=domain, input_dim=input_dim, output_dim=output_dim)
		self.funcs = list(funcs)

	def _eval(self, pos_arr):
		return gen_sum([f._eval(pos_arr) for f in self.funcs])

	def derivative(self, *args, **kwargs):
		return gen_sum([f.derivative(*args, **kwargs) for f in self.funcs])

	def __add__(self, other):
		if isinstance(other, ZeroFunc):
			return self
		if isinstance(other, FuncBase):
			extra = other.funcs if isinstance(other, SumOfFuncs) else [other]
			return SumOfFuncs(self.funcs + extra, _key=_SENTINEL)
		return super().__add__(other)

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		if isinstance(other, ZeroFunc):
			od = next((d for d in (other.output_dim, self.output_dim) if d is not None), 1)
			return ZeroFunc(domain=lambda pos: self.domain(pos) and other.domain(pos), input_dim=self.input_dim, output_dim=od)
		if np.isscalar(other) or (isinstance(other, ConstFunc) and other.output_dim == 1):
			val = other if np.isscalar(other) else other.const
			return SumOfFuncs([val * f for f in self.funcs], _key=_SENTINEL)
		if isinstance(other, FuncBase):
			return ProdOfFuncs([self, other], _key=_SENTINEL)
		return NotImplemented

	def __rmul__(self, other):
		return self.__mul__(other)

	def simplify(self, deep=False):
		funcs = list(self.funcs)
		if deep:
			funcs = [f.simplify(deep=True) if hasattr(f, 'simplify') else f for f in funcs]
		changed = True
		while changed:
			changed = False
			i = 0
			while i < len(funcs):
				j = i + 1
				while j < len(funcs):
					result = funcs[i] + funcs[j]
					if not isinstance(result, SumOfFuncs):
						funcs[i] = result
						funcs.pop(j)
						changed = True
					else:
						j += 1
				i += 1
		return gen_sum(funcs)

	def _gradient_component(self, coord):
		results = [f._gradient_component(coord) if hasattr(f, '_gradient_component') else NotImplemented for f in self.funcs]
		if any(r is NotImplemented for r in results):
			return NotImplemented
		return gen_sum(results)

	def integrate(self, a=None, b=None):
		if a is None and b is None:
			return gen_sum([f.integrate() for f in self.funcs])
		return sum(f.integrate(a, b) for f in self.funcs)

	def copy(self):
		return SumOfFuncs([f.copy() for f in self.funcs], _key=_SENTINEL)

	def sympy_output(self):
		return gen_sum([f.sympy_output() for f in self.funcs])


class ProdOfFuncs(FuncBase):
	# Same coord_sys caveat as SumOfFuncs: no natural coordinate system. Functions from different
	# coordinate systems combined with * will silently produce an object whose gradient() and
	# derivative() calls may fail or give wrong results.

	def __new__(cls, funcs, _key=None):
		if _key is not _SENTINEL:
			raise TypeError("ProdOfFuncs is internal — combine functions with * instead.")
		if len(funcs) == 0:
			return ConstFunc(1)
		if len(funcs) == 1:
			return funcs[0]
		return object.__new__(cls)

	def __init__(self, funcs, _key=None):
		input_dim, output_dim = _check_dim_compat(funcs, 'mul')
		domain = lambda pos: all(f.domain(pos) for f in funcs)
		super().__init__(domain=domain, input_dim=input_dim, output_dim=output_dim)
		self.funcs = list(funcs)

	def _eval(self, pos_arr):
		return gen_prod([f._eval(pos_arr) for f in self.funcs])

	def derivative(self, *args, **kwargs):
		if any(getattr(f, 'input_dim', None) == 0 for f in self.funcs):
			raise TypeError(
				"ProdOfFuncs uses the continuous product rule, which is incorrect for "
				"discrete functions. Use concrete subclass __mul__ for analytic products "
				"(e.g. ExpSeq * ExpSeq → single ExpSeq)."
			)
		terms = []
		for i, fi in enumerate(self.funcs):
			others = self.funcs[:i] + self.funcs[i+1:]
			term = gen_prod([fi.derivative(*args, **kwargs)] + others)
			terms.append(term)
		return gen_sum(terms)

	def __mul__(self, other):
		if np.isscalar(other) or (isinstance(other, ConstFunc) and other.output_dim == 1):
			val = other if np.isscalar(other) else other.const
			return ProdOfFuncs([val * self.funcs[0]] + self.funcs[1:], _key=_SENTINEL)
		if isinstance(other, ProdOfFuncs):
			return ProdOfFuncs(self.funcs + other.funcs, _key=_SENTINEL)
		if isinstance(other, FuncBase):
			return ProdOfFuncs(self.funcs + [other], _key=_SENTINEL)
		return NotImplemented

	def __rmul__(self, other):
		return self.__mul__(other)

	def __add__(self, other):
		if isinstance(other, FuncBase):
			return SumOfFuncs([self, other], _key=_SENTINEL)
		return super().__add__(other)

	def reciprocal(self):
		return ProdOfFuncs([f.reciprocal() for f in self.funcs], _key=_SENTINEL)

	def simplify(self, deep=False):
		funcs = list(self.funcs)
		if deep:
			funcs = [f.simplify(deep=True) if hasattr(f, 'simplify') else f for f in funcs]
		changed = True
		while changed:
			changed = False
			i = 0
			while i < len(funcs):
				j = i + 1
				while j < len(funcs):
					result = funcs[i] * funcs[j]
					if not isinstance(result, ProdOfFuncs):
						funcs[i] = result
						funcs.pop(j)
						changed = True
					else:
						j += 1
				i += 1
		return gen_prod(funcs)

	def copy(self):
		return ProdOfFuncs([f.copy() for f in self.funcs], _key=_SENTINEL)

	def sympy_output(self):
		return gen_prod([f.sympy_output() for f in self.funcs])


class RatioOfFuncs(FuncBase):

	def __new__(cls, numer, denom, _key=None):
		if _key is not _SENTINEL:
			raise TypeError("RatioOfFuncs is internal — combine functions with / instead.")
		if isinstance(numer, ZeroFunc):
			od = numer.output_dim if numer.output_dim is not None else 1
			return ZeroFunc(domain=lambda pos: numer.domain(pos) and denom.domain(pos),
							input_dim=numer.input_dim, output_dim=od)
		if isinstance(denom, ZeroFunc):
			raise ZeroDivisionError("Cannot divide by ZeroFunc")
		if isinstance(denom, ConstFunc):
			return (1.0 / denom.const) * numer
		return object.__new__(cls)

	def __init__(self, numer, denom, _key=None):
		input_dim, output_dim = _check_dim_compat([numer, denom], 'mul')
		domain = lambda pos: numer.domain(pos) and denom.domain(pos)
		super().__init__(domain=domain, input_dim=input_dim, output_dim=output_dim)
		self.numer = numer
		self.denom = denom

	def _eval(self, pos_arr):
		n = self.numer._eval(pos_arr)
		d = self.denom._eval(pos_arr)
		if n.ndim == 1 and d.ndim == 2:
			n = n[:, np.newaxis]
		elif n.ndim == 2 and d.ndim == 1:
			d = d[:, np.newaxis]
		return n / d

	def derivative(self, *args, **kwargs):
		f, g = self.numer, self.denom
		return RatioOfFuncs(
			f.derivative(*args, **kwargs) * g - f * g.derivative(*args, **kwargs),
			ProdOfFuncs([g, g], _key=_SENTINEL),
			_key=_SENTINEL,
		)

	def reciprocal(self):
		return RatioOfFuncs(self.denom, self.numer, _key=_SENTINEL)

	def sympy_output(self):
		return self.numer.sympy_output() / self.denom.sympy_output()

	def copy(self):
		return RatioOfFuncs(self.numer.copy(), self.denom.copy(), _key=_SENTINEL)


class ComposedFunc(FuncBase):

	def __new__(cls, outer, inner):
		if inner.output_dim is not None and outer.input_dim is not None and inner.output_dim != outer.input_dim:
			raise ValueError(f"ComposedFunc: inner.output_dim={inner.output_dim} does not match outer.input_dim={outer.input_dim}")
		composed_domain = lambda pos: inner.domain(pos) and outer.domain(inner(pos, ignore_domain=True))
		if isinstance(outer, ZeroFunc):
			od = outer.output_dim if outer.output_dim is not None else 1
			return ZeroFunc(domain=composed_domain, input_dim=inner.input_dim, output_dim=od)
		if isinstance(outer, ConstFunc):
			return ConstFunc(outer.const, domain=composed_domain, input_dim=inner.input_dim, output_dim=outer.output_dim)
		if isinstance(inner, ZeroFunc) and inner.output_dim == 1:
			return ConstFunc(outer(0.0, ignore_domain=True), domain=composed_domain, input_dim=inner.input_dim, output_dim=outer.output_dim)
		if isinstance(inner, ConstFunc) and inner.output_dim == 1:
			return ConstFunc(outer(inner.const, ignore_domain=True), domain=composed_domain, input_dim=inner.input_dim, output_dim=outer.output_dim)
		return object.__new__(cls)

	def __init__(self, outer, inner):
		domain = lambda pos: inner.domain(pos) and outer.domain(inner(pos, ignore_domain=True))
		super().__init__(domain=domain, input_dim=inner.input_dim, output_dim=outer.output_dim)
		self.outer = outer
		self.inner = inner

	def _eval(self, pos_arr):
		return self.outer._eval(self.inner._eval(pos_arr))

	def copy(self):
		return ComposedFunc(self.outer.copy(), self.inner.copy())

	def derivative(self, *args, **kwargs):
		if self.outer.input_dim == 1:
			return ComposedFunc(self.outer.derivative(), self.inner) * self.inner.derivative(*args, **kwargs)
		raise NotImplementedError("Derivative of ComposedFunc not implemented for non-scalar outer function.")

	def sympy_output(self):
		if self.outer.input_dim == 1:
			return self.outer.sympy_output().subs(self.outer.sympy_var, self.inner.sympy_output())
		raise NotImplementedError("sympy_output not implemented for non-1D ComposedFunc")

class VecFunc(FuncBase):
	def __init__(self, components):
		input_dim, _ = _check_dim_compat([*components, ConstFunc(1)], 'sum')
		super().__init__(domain=lambda pos: all(f.domain(pos) for f in components),
						 input_dim=input_dim, output_dim=len(components))
		self.components = list(components) # TODO: I always have word wrap on, so these line breaks are unnecessary.

	def _eval(self, pos_arr):
		return np.stack([f._eval(pos_arr) for f in self.components], axis=1)  # (N, k)

	def derivative(self, *args, **kwargs):
		return VecFunc([f.derivative(*args, **kwargs) for f in self.components])

	def sympy_output(self):
		return sym.Matrix([f.sympy_output() for f in self.components])

	def copy(self):
		return VecFunc([f.copy() for f in self.components])

	def __add__(self, other):
		if isinstance(other, ZeroFunc):
			_check_dim_compat([self, other], 'sum')
			return self
		if isinstance(other, VecFunc):
			if len(self.components) != len(other.components):
				raise ValueError(f"Cannot add VecFuncs with different lengths: {len(self.components)} vs {len(other.components)}")
			return VecFunc([f + g for f, g in zip(self.components, other.components)])
		return super().__add__(other)

	def __mul__(self, other):
		if isinstance(other, ZeroFunc):
			input_dim, output_dim = _check_dim_compat([self, other], 'mul')
			if output_dim is None:
				output_dim = 1
			return ZeroFunc(domain=lambda pos: self.domain(pos) and other.domain(pos), input_dim=input_dim, output_dim=output_dim)
		if np.isscalar(other):
			return VecFunc([other * f for f in self.components])
		if isinstance(other, VecFunc):
			if len(self.components) != len(other.components):
				raise ValueError(f"Cannot multiply VecFuncs with different lengths: {len(self.components)} vs {len(other.components)}")
			return VecFunc([f * g for f, g in zip(self.components, other.components)])
		if isinstance(other, FuncBase) and (other.output_dim is None or other.output_dim == 1):
			return VecFunc([other * f for f in self.components])
		arr = np.asarray(other)
		if arr.ndim == 1:
			if len(arr) != len(self.components):
				raise ValueError(f"Cannot multiply VecFunc({len(self.components)}) by length-{len(arr)} array")
			return VecFunc([c * f for c, f in zip(arr, self.components)])
		return NotImplemented

	def __getitem__(self, i):
		return self.components[i]

	def simplify(self, deep=False):
		return VecFunc([f.simplify(deep=deep) if hasattr(f, 'simplify') else f for f in self.components])

######### Coordinate transformation functions
class CoordPow(FuncBase):
	"""pos[coord_index]^power for any input_dim — used as coordinate scale factors in CoordSystem."""

	def __new__(cls, input_dim, coord_index, power=1, coord_name=None, domain=lambda _: True):
		if power == 0:
			return ConstFunc(1, domain=domain, input_dim=input_dim, output_dim=1)
		return object.__new__(cls)

	def __init__(self, input_dim, coord_index, power=1, coord_name=None, domain=lambda _: True):
		super().__init__(domain=domain, input_dim=input_dim, output_dim=1)
		self.coord_index = coord_index
		self.power = power
		self.coord_name = coord_name if coord_name is not None else f'x{coord_index}'

	def _eval(self, pos_arr):
		col = pos_arr if self.input_dim == 1 else pos_arr[:, self.coord_index]
		return col ** self.power

	def derivative(self, coord, *args, **kwargs):
		if coord == self.coord_name:
			return self.power * CoordPow(self.input_dim, self.coord_index, self.power - 1, self.coord_name, self.domain)
		return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=1)

	def reciprocal(self):
		return CoordPow(self.input_dim, self.coord_index, -self.power, self.coord_name, self.domain)

	def sympy_output(self):
		return sym.Symbol(self.coord_name, real=True) ** self.power

	def copy(self):
		return CoordPow(self.input_dim, self.coord_index, self.power, self.coord_name, self.domain)


class TrigCoord(FuncBase):
	"""sin, cos, csc, or sec of one coordinate of a multi-dimensional input."""

	_inv = {'sin': 'csc', 'cos': 'sec', 'csc': 'sin', 'sec': 'cos'}

	def __init__(self, func, input_dim, coord_index, coord_name=None, domain=lambda _: True):
		super().__init__(domain=domain, input_dim=input_dim, output_dim=1)
		if func not in self._inv:
			raise ValueError(f"TrigCoord func must be one of {list(self._inv)}, got '{func}'")
		self.func = func
		self.coord_index = coord_index
		self.coord_name = coord_name if coord_name is not None else f'x{coord_index}'

	def _eval(self, pos_arr):
		col = pos_arr[:, self.coord_index]
		if self.func == 'sin': return np.sin(col)
		if self.func == 'cos': return np.cos(col)
		if self.func == 'csc': return 1.0 / np.sin(col)
		if self.func == 'sec': return 1.0 / np.cos(col)
		raise AssertionError(f"Unreachable: unknown func {self.func!r}")

	def derivative(self, coord, *args, **kwargs):
		if coord != self.coord_name:
			return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=1)
		t = lambda f: TrigCoord(f, self.input_dim, self.coord_index, self.coord_name, self.domain)
		if self.func == 'sin': return t('cos')
		if self.func == 'cos': return ConstFunc(-1, input_dim=self.input_dim, output_dim=1) * t('sin')
		if self.func == 'csc': return ConstFunc(-1, input_dim=self.input_dim, output_dim=1) * t('csc') * t('csc') * t('cos')
		if self.func == 'sec': return t('sec') * t('sec') * t('sin')
		raise AssertionError(f"Unreachable: unknown func {self.func!r}")

	def reciprocal(self):
		return TrigCoord(self._inv[self.func], self.input_dim, self.coord_index, self.coord_name, self.domain)

	def sympy_output(self):
		s = sym.Symbol(self.coord_name, real=True)
		if self.func == 'sin': return sym.sin(s)
		if self.func == 'cos': return sym.cos(s)
		if self.func == 'csc': return sym.csc(s)
		if self.func == 'sec': return sym.sec(s)
		raise AssertionError(f"Unreachable: unknown func {self.func!r}")

	def copy(self):
		return TrigCoord(self.func, self.input_dim, self.coord_index, self.coord_name, self.domain)

# Helper functions
def gen_sum(l, start=None):
	if start is None:
		start = 0
	if len(l) == 0:
		return start
	return sum(l[1:], l[0])


def gen_prod(l, start=None):
	if start is None:
		start = 1
	if len(l) == 0:
		return start
	result = l[0]
	for item in l[1:]:
		result = result * item
	return result