import math
import numpy as np


_SENTINEL = object()  # Used to control internal constructor behavior of the combinator classes


def _check_dim_compat(funcs, mode):
	in_dims = {f.input_dim for f in funcs}
	if len(in_dims) > 1:
		raise ValueError(f"Cannot combine functions with different input_dim: {in_dims}")
	input_dim = in_dims.pop()

	out_dims = [f.output_dim for f in funcs]
	if mode == 'sum':
		known = set(out_dims)
		if len(known) > 1:
			raise ValueError(f"incompatible output_dim for addition: {known}")
		output_dim = known.pop()
	else:  # 'mul'
		known = {d for d in out_dims if d != 1}
		if len(known) > 1:
			raise ValueError(f"incompatible output_dim for multiplication: {known}")
		output_dim = known.pop() if known else 1

	return input_dim, output_dim


class FuncBase:

	def __init__(self, domain=lambda _: True, input_dim=None, output_dim=None):
		if input_dim is None:
			raise ValueError("input_dim must be specified")
		if output_dim is None:
			raise ValueError("output_dim must be specified")
		self.domain = domain
		self.input_dim = input_dim
		self.output_dim = output_dim

	def pos_to_arr(self, pos):
		d = self.input_dim
		pos_arr = np.asarray(pos)
		if np.issubdtype(pos_arr.dtype, np.complexfloating):
			raise ValueError(f"Inputs must be real-valued, got dtype {pos_arr.dtype}. Complex inputs are not supported.")
		pos_arr = pos_arr.astype(float)
		if d == 1:
			if pos_arr.ndim == 0:
				return pos_arr.reshape(1), True
			if pos_arr.ndim == 1:
				return pos_arr, False
			raise ValueError("1D function expects a scalar or 1D array")
		if pos_arr.shape == (d,):
			return pos_arr.reshape(1, d), True
		if pos_arr.ndim == 2 and pos_arr.shape[1] == d:
			return pos_arr, False
		raise ValueError(f"{d}D function expects a {d}-vector or an (N, {d}) array")


	def _eval(self, pos_arr):
		raise NotImplementedError(f"{type(self).__name__} must implement _eval")

	def derivative(self, *args, **kwargs):
		raise NotImplementedError(f"{type(self).__name__} must implement derivative")

	def antiderivative(self):
		raise NotImplementedError(f"{type(self).__name__} must implement antiderivative")

	def sympy_output(self):
		raise NotImplementedError(f"{type(self).__name__} must implement sympy_output")

	def clone(self, **overrides):
		"""Public: rebuild an instance of the same class with any constructor kwargs overridden —
		``f.clone(ampl=5)`` is a copy-with-changes (cf. ``dataclasses.replace``). It is also the
		single customization point for reconstruction, used by ``copy`` and by leaf classes for
		concise derivatives (e.g. ``self.clone(ampl=...)``). Leaf classes declare
		``self.parameters`` (constructor kwargs) in __init__ and inherit this default; the
		combinators (``SumOfFuncs``, ``ProdOfFuncs``, ``RatioOfFuncs``, ``ComposedFunc``,
		``VecFunc``) hold sub-functions instead of flat parameters, so they override ``clone`` to
		deep-copy (or override) those sub-functions."""
		return type(self)(**{**self.parameters, **overrides})

	def copy(self):
		"""Conventional no-arg deep-ish copy; an alias for ``self.clone()``."""
		return self.clone()

	def __repr__(self):
		return (f"<{type(self).__name__}>"
				" (call .sympy_output() for a symbolic representation — may be slow for large expressions)")

	def numerical_derivative(self, pos, coord_index=0, eps=1e-5):
		pos_arr, single_input = self.pos_to_arr(pos)
		if self.input_dim == 1:
			eps_step = eps
		else:
			eps_step = np.zeros(self.input_dim)
			eps_step[coord_index] = eps
		result = (self(pos_arr + eps_step, ignore_domain=True) - self(pos_arr - eps_step, ignore_domain=True)) / (2 * eps)
		return result[0] if single_input else result

	def integrate(self, a=None, b=None):
		"""Indefinite integral (no args) via ``antiderivative``; definite integral over [a, b]
		via the antiderivative's endpoints. Subclasses with special handling (numerical fallback,
		term-wise definite integrals) override this."""
		if a is None and b is None:
			return self.antiderivative()
		if a is not None and b is not None:
			F = self.antiderivative()
			return F(b, ignore_domain=True) - F(a, ignore_domain=True)
		raise ValueError(
			"integrate() takes either no positional arguments (indefinite) "
			"or exactly two (definite interval [a, b])"
		)

	def reciprocal(self):
		if self.output_dim > 1:
			numer = ConstFunc(np.ones(self.output_dim), input_dim=self.input_dim)
		else:
			numer = ConstFunc(1, input_dim=self.input_dim)
		return RatioOfFuncs(numer, self, _key=_SENTINEL)

	def _resolve_coord_sys(self, coord_sys):
		"""Resolve the coordinate system for a differential operator: explicit arg, else the
		function's own ``coord_sys`` (a class attribute on Funcs3D subclasses)."""
		if coord_sys is None:
			coord_sys = getattr(self, 'coord_sys', None)
		if coord_sys is None:
			raise ValueError(f"{type(self).__name__} has no natural coordinate system; pass coord_sys explicitly")
		return coord_sys

	def _gradient_component(self, coord):
		"""Hook for the ``coord``-component of the gradient. Default: NotImplemented, so
		``gradient`` falls back to ``inv_h * derivative(coord)``. Override to absorb scale
		factors analytically (e.g. Cylindrical absorbs 1/rho via Bessel recurrences)."""
		return NotImplemented

	def gradient(self, coord_sys=None):
		coord_sys = self._resolve_coord_sys(coord_sys)
		if self.input_dim != len(coord_sys.coords):
			raise ValueError(f"gradient: function input_dim={self.input_dim} does not match coordinate system dimension {len(coord_sys.coords)}")
		if self.output_dim != 1:
			raise ValueError(f"gradient: function output_dim={self.output_dim} must be 1 (scalar field)")
		components = []
		for coord, inv_h in zip(coord_sys.coords, coord_sys.inv_scale_factors):
			result = self._gradient_component(coord)
			if result is NotImplemented:
				df = self.derivative(coord)
				result = df if (isinstance(inv_h, ConstFunc) and inv_h.const == 1) else inv_h * df
			components.append(result)
		return VecFunc(components)

	def divergence(self, coord_sys=None):
		"""Divergence of a vector field. Defined for any function whose output_dim equals the
		coordinate-system dimension (a VecFunc, or a vector-valued Funcs3D with output_dim=3, …).
		Components are read via ``self[i]``, so the function must support indexing."""
		coord_sys = self._resolve_coord_sys(coord_sys)
		n = len(coord_sys.coords)
		if self.output_dim != n:
			raise ValueError(f"divergence: vector field output_dim={self.output_dim} does not match coordinate system dimension {n}")
		if self.input_dim != n:
			raise ValueError(f"divergence: vector field input_dim={self.input_dim} does not match coordinate system dimension {n}")
		scale_factors = [inv_h.reciprocal() for inv_h in coord_sys.inv_scale_factors]
		terms = []
		for i, coord in enumerate(coord_sys.coords):
			fi = self[i]
			others_h = [scale_factors[j] for j in range(n) if j != i and not (isinstance(scale_factors[j], ConstFunc) and scale_factors[j].const == 1)]
			coeff_fi = math.prod(others_h + [fi])
			terms.append(coeff_fi.derivative(coord))
		total = sum(terms)
		nontrivial_inv = [inv_h for inv_h in coord_sys.inv_scale_factors if not (isinstance(inv_h, ConstFunc) and inv_h.const == 1)]
		result = total
		for inv_h in nontrivial_inv:
			result = inv_h * result
		return result

	def laplacian(self, coord_sys=None):
		coord_sys = self._resolve_coord_sys(coord_sys)
		return self.gradient(coord_sys).divergence(coord_sys)

	def __call__(self, pos, ignore_domain=False):
		pos_arr, single_input = self.pos_to_arr(pos)
		if not ignore_domain:
			self._check_domain(pos_arr)
		result = self._eval(pos_arr)
		return result[0] if single_input else result

	def _check_domain(self, pos_arr, output='error'):
		"""Domain check on an already-normalised pos_arr (N, d) or (N,) array."""
		bad_points = [p for p in pos_arr if not self.domain(p)]
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
		val = as_scalar(other)
		if val is not None:
			if val == 0:
				return self
			return SumOfFuncs([self, ConstFunc(val, input_dim=self.input_dim)], _key=_SENTINEL)
		if isinstance(other, FuncBase):
			return SumOfFuncs([self, other], _key=_SENTINEL)
		arr = np.asarray(other)
		if arr.ndim == 1 and len(arr) == self.output_dim:
			return SumOfFuncs([self, ConstFunc(arr, input_dim=self.input_dim)], _key=_SENTINEL)
		return NotImplemented

	def __radd__(self, other):
		return self.__add__(other)

	def __mul__(self, other):
		# Non-FuncBase operands: a numeric scalar gives identities (1 → self, 0 → ZeroFunc) or
		# folds into a ConstFunc; a 1-D array scales component-wise; anything else is unknown.
		if not isinstance(other, FuncBase):
			val = as_scalar(other)
			if val is not None:
				if val == 1:
					return self
				if val == 0:
					return ZeroFunc(domain=self.domain, input_dim=self.input_dim, output_dim=self.output_dim)
				return ProdOfFuncs([ConstFunc(val, input_dim=self.input_dim), self], _key=_SENTINEL)
			arr = np.asarray(other)
			if arr.ndim == 1:
				if len(arr) != self.output_dim:
					raise ValueError(f"Cannot multiply output_dim={self.output_dim} function by length-{len(arr)} array")
				return ProdOfFuncs([ConstFunc(arr, input_dim=self.input_dim), self], _key=_SENTINEL)
			return NotImplemented
		# FuncBase operands: require compatible dimensions before combining. ZeroFunc absorbs
		# to a ZeroFunc of the *combined* shape; everything else becomes a ProdOfFuncs.
		input_dim, output_dim = _check_dim_compat([self, other], 'mul')
		if isinstance(other, ZeroFunc):
			return ZeroFunc(domain=combine_domains(self, other), input_dim=input_dim, output_dim=output_dim)
		return ProdOfFuncs([self, other], _key=_SENTINEL)

	def __rmul__(self, other):
		return self.__mul__(other)

	def __truediv__(self, other):
		val = as_scalar(other)
		if val is not None:
			return self * (1.0 / val)
		if isinstance(other, FuncBase):
			return RatioOfFuncs(self, other, _key=_SENTINEL)
		return NotImplemented

	def __rtruediv__(self, other):
		val = as_scalar(other)
		if val is not None:
			return RatioOfFuncs(ConstFunc(val, input_dim=self.input_dim), self, _key=_SENTINEL)
		return NotImplemented

	def __eq__(self, other):
		try:
			diff = self - other
		except Exception:
			return NotImplemented
		if not isinstance(diff, FuncBase):
			return NotImplemented
		if hasattr(diff, 'simplify'):
			diff = diff.simplify()
		return isinstance(diff, ZeroFunc)

	# __eq__ is value-based and potentially expensive, so these objects are not hashable.
	__hash__ = None


# Helper functions
def as_scalar(x):
	"""Return a Python scalar if x is a numeric scalar (incl. a 0-d ndarray), else None. Never
	unwraps a FuncBase — those go through the FuncBase code paths."""
	if isinstance(x, FuncBase):
		return None
	if np.isscalar(x):
		return x
	arr = np.asarray(x)
	if arr.ndim == 0 and np.issubdtype(arr.dtype, np.number):
		return arr.item()
	return None


def scalar_factor(x):
	"""Scalar value if x is a numeric scalar or a scalar (output_dim==1) ConstFunc, else None.
	Used by leaf classes to fold a scalar multiplier into a parameter (dropping any ConstFunc
	domain, matching the historical behaviour)."""
	if isinstance(x, ConstFunc) and x.output_dim == 1:
		return x.const
	return as_scalar(x)


def combine_domains(*items):
	"""Domain that holds where all items' domains hold. Each item may be a FuncBase (its .domain
	is used) or a raw domain callable."""
	domains = [it.domain if isinstance(it, FuncBase) else it for it in items]
	return lambda pos: all(d(pos) for d in domains)


def _fuse_pairwise(funcs, op, container_cls):
	"""Repeatedly combine pairs via ``op``; accept a combination only when it is no longer a
	``container_cls`` (i.e. the two terms genuinely fused). Shared by SumOfFuncs/ProdOfFuncs."""
	funcs = list(funcs)
	changed = True
	while changed:
		changed = False
		i = 0
		while i < len(funcs):
			j = i + 1
			while j < len(funcs):
				result = op(funcs[i], funcs[j])
				if not isinstance(result, container_cls):
					funcs[i] = result
					funcs.pop(j)
					changed = True
				else:
					j += 1
			i += 1
	return funcs


# FuncBase's operators construct the concrete leaf/combinator classes by name at call time. The
# imports are deferred to module-bottom to break the import cycle (leaves/combinators subclass
# FuncBase): by the time any method runs, these names are bound as module globals.
from .leaves import ZeroFunc, ConstFunc
from .combinators import SumOfFuncs, ProdOfFuncs, RatioOfFuncs, VecFunc
