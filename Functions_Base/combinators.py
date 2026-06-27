import math
import operator
import numpy as np
import sympy as sym

from .func_base import FuncBase, _check_dim_compat, combine_domains, scalar_factor, _fuse_pairwise, _SENTINEL
from .leaves import ZeroFunc, ConstFunc

######### Different ways to combine functions

class SumOfFuncs(FuncBase):
	# gradient()/laplacian() without an explicit coord_sys will raise ValueError here since SumOfFuncs
	# has no coord_sys attribute. If all terms share the same coord_sys, pass it explicitly. Terms from
	# different coordinate systems will also fail in gradient() at the derivative() call.

	def __new__(cls, funcs, _key=None):
		if _key is not _SENTINEL:
			raise TypeError("SumOfFuncs is internal — combine functions with + instead.")
		if len(funcs) == 0:
			raise ValueError("SumOfFuncs requires at least one function")
		if len(funcs) == 1:
			return funcs[0]
		return object.__new__(cls)

	def __init__(self, funcs, _key=None):
		input_dim, output_dim = _check_dim_compat(funcs, 'sum')
		super().__init__(domain=combine_domains(*funcs), input_dim=input_dim, output_dim=output_dim)
		self.funcs = list(funcs)

	def _eval(self, pos_arr):
		return sum(f._eval(pos_arr) for f in self.funcs)

	def derivative(self, *args, **kwargs):
		return SumOfFuncs([f.derivative(*args, **kwargs) for f in self.funcs], _key=_SENTINEL)

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
		# Only specialise scalar multiplication (distribute over the terms); ZeroFunc
		# absorption, generic FuncBase → ProdOfFuncs, and array handling all defer to
		# FuncBase.__mul__.
		val = scalar_factor(other)
		if val is not None:
			return SumOfFuncs([val * f for f in self.funcs], _key=_SENTINEL)
		return super().__mul__(other)

	def simplify(self, deep=False):
		funcs = [f.simplify(deep=True) if hasattr(f, 'simplify') else f for f in self.funcs] if deep else list(self.funcs)
		funcs = _fuse_pairwise(funcs, operator.add, SumOfFuncs)
		return SumOfFuncs(funcs, _key=_SENTINEL)

	def _gradient_component(self, coord):
		results = [f._gradient_component(coord) for f in self.funcs]
		if any(r is NotImplemented for r in results):
			return NotImplemented
		return SumOfFuncs(results, _key=_SENTINEL)

	def antiderivative(self):
		return SumOfFuncs([f.antiderivative() for f in self.funcs], _key=_SENTINEL)

	def clone(self, **overrides):
		funcs = overrides.get('funcs', [f.copy() for f in self.funcs])
		return SumOfFuncs(funcs, _key=_SENTINEL)

	def sympy_output(self):
		return sum(f.sympy_output() for f in self.funcs)


class ProdOfFuncs(FuncBase):
	# Same coord_sys caveat as SumOfFuncs: no natural coordinate system. Functions from different
	# coordinate systems combined with * will silently produce an object whose gradient() and
	# derivative() calls may fail or give wrong results.

	def __new__(cls, funcs, _key=None):
		if _key is not _SENTINEL:
			raise TypeError("ProdOfFuncs is internal — combine functions with * instead.")
		if len(funcs) == 0:
			raise ValueError("ProdOfFuncs requires at least one function")
		if len(funcs) == 1:
			return funcs[0]
		return object.__new__(cls)

	def __init__(self, funcs, _key=None):
		input_dim, output_dim = _check_dim_compat(funcs, 'mul')
		super().__init__(domain=combine_domains(*funcs), input_dim=input_dim, output_dim=output_dim)
		self.funcs = list(funcs)

	def _eval(self, pos_arr):
		evals = [f._eval(pos_arr) for f in self.funcs]
		# Mixed output dims (a scalar factor times a vector factor): promote each scalar
		# (N,) result to (N, 1) so it broadcasts component-wise against the (N, k) vector,
		# rather than along the sample axis. (RatioOfFuncs._eval does the same for /.)
		if any(e.ndim == 2 for e in evals):
			evals = [e[:, np.newaxis] if e.ndim == 1 else e for e in evals]
		return math.prod(evals)

	def derivative(self, *args, **kwargs):
		terms = []
		for i, fi in enumerate(self.funcs):
			others = self.funcs[:i] + self.funcs[i+1:]
			term = ProdOfFuncs([fi.derivative(*args, **kwargs)] + others, _key=_SENTINEL)
			terms.append(term)
		return SumOfFuncs(terms, _key=_SENTINEL)

	def __mul__(self, other):
		# Defer ZeroFunc absorption to FuncBase.__mul__ so the product collapses to a
		# ZeroFunc rather than swallowing the zero as an extra factor; specialise only the
		# scalar fold and the flattening of FuncBase factors into this product.
		if isinstance(other, ZeroFunc):
			return super().__mul__(other)
		val = scalar_factor(other)
		if val is not None:
			return ProdOfFuncs([val * self.funcs[0]] + self.funcs[1:], _key=_SENTINEL)
		if isinstance(other, ProdOfFuncs):
			return ProdOfFuncs(self.funcs + other.funcs, _key=_SENTINEL)
		if isinstance(other, FuncBase):
			return ProdOfFuncs(self.funcs + [other], _key=_SENTINEL)
		return NotImplemented

	def __add__(self, other):
		if isinstance(other, FuncBase):
			return SumOfFuncs([self, other], _key=_SENTINEL)
		return super().__add__(other)

	def reciprocal(self):
		return ProdOfFuncs([f.reciprocal() for f in self.funcs], _key=_SENTINEL)

	def simplify(self, deep=False):
		funcs = [f.simplify(deep=True) if hasattr(f, 'simplify') else f for f in self.funcs] if deep else list(self.funcs)
		funcs = _fuse_pairwise(funcs, operator.mul, ProdOfFuncs)
		return ProdOfFuncs(funcs, _key=_SENTINEL)

	def clone(self, **overrides):
		funcs = overrides.get('funcs', [f.copy() for f in self.funcs])
		return ProdOfFuncs(funcs, _key=_SENTINEL)

	def sympy_output(self):
		return math.prod(f.sympy_output() for f in self.funcs)


class RatioOfFuncs(FuncBase):

	def __new__(cls, numer, denom, _key=None):
		if _key is not _SENTINEL:
			raise TypeError("RatioOfFuncs is internal — combine functions with / instead.")
		if isinstance(numer, ZeroFunc):
			return ZeroFunc(domain=combine_domains(numer, denom),
							input_dim=numer.input_dim, output_dim=numer.output_dim)
		if isinstance(denom, ZeroFunc):
			raise ZeroDivisionError("Cannot divide by ZeroFunc")
		if isinstance(denom, ConstFunc):
			return (1.0 / denom.const) * numer
		return object.__new__(cls)

	def __init__(self, numer, denom, _key=None):
		input_dim, output_dim = _check_dim_compat([numer, denom], 'mul')
		super().__init__(domain=combine_domains(numer, denom), input_dim=input_dim, output_dim=output_dim)
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

	def clone(self, **overrides):
		numer = overrides.get('numer', self.numer.copy())
		denom = overrides.get('denom', self.denom.copy())
		return RatioOfFuncs(numer, denom, _key=_SENTINEL)


class ComposedFunc(FuncBase):

	def __new__(cls, outer, inner):
		if inner.output_dim != outer.input_dim:
			raise ValueError(f"ComposedFunc: inner.output_dim={inner.output_dim} does not match outer.input_dim={outer.input_dim}")
		composed_domain = lambda pos: inner.domain(pos) and outer.domain(inner(pos, ignore_domain=True))
		if isinstance(outer, ZeroFunc):
			return ZeroFunc(domain=composed_domain, input_dim=inner.input_dim, output_dim=outer.output_dim)
		if isinstance(outer, ConstFunc):
			return ConstFunc(outer.const, domain=composed_domain, input_dim=inner.input_dim)
		if isinstance(inner, ZeroFunc) and inner.output_dim == 1:
			return ConstFunc(outer(0.0, ignore_domain=True), domain=composed_domain, input_dim=inner.input_dim)
		if isinstance(inner, ConstFunc) and inner.output_dim == 1:
			return ConstFunc(outer(inner.const, ignore_domain=True), domain=composed_domain, input_dim=inner.input_dim)
		return object.__new__(cls)

	def __init__(self, outer, inner):
		domain = lambda pos: inner.domain(pos) and outer.domain(inner(pos, ignore_domain=True))
		super().__init__(domain=domain, input_dim=inner.input_dim, output_dim=outer.output_dim)
		self.outer = outer
		self.inner = inner

	def _eval(self, pos_arr):
		return self.outer._eval(self.inner._eval(pos_arr))

	def clone(self, **overrides):
		outer = overrides.get('outer', self.outer.copy())
		inner = overrides.get('inner', self.inner.copy())
		return ComposedFunc(outer, inner)

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
		if not components:
			raise ValueError("VecFunc requires at least one component")
		non_scalar = [f for f in components if f.output_dim != 1]
		if non_scalar:
			raise ValueError(f"All VecFunc components must be scalar (output_dim=1)")
		input_dim, _ = _check_dim_compat(components, 'sum')
		super().__init__(domain=combine_domains(*components),
						 input_dim=input_dim, output_dim=len(components))
		self.components = list(components)

	def _eval(self, pos_arr):
		return np.stack([f._eval(pos_arr) for f in self.components], axis=1)  # (N, k)

	def derivative(self, *args, **kwargs):
		return VecFunc([f.derivative(*args, **kwargs) for f in self.components])

	def sympy_output(self):
		return sym.Matrix([f.sympy_output() for f in self.components])

	def clone(self, **overrides):
		components = overrides.get('components', [f.copy() for f in self.components])
		return VecFunc(components)

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
			return ZeroFunc(domain=combine_domains(self, other), input_dim=input_dim, output_dim=output_dim)
		if np.isscalar(other):
			return VecFunc([other * f for f in self.components])
		if isinstance(other, VecFunc):
			if len(self.components) != len(other.components):
				raise ValueError(f"Cannot multiply VecFuncs with different lengths: {len(self.components)} vs {len(other.components)}")
			return VecFunc([f * g for f, g in zip(self.components, other.components)])
		if isinstance(other, FuncBase) and other.output_dim == 1:
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
