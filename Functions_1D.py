
import numpy as np
import sympy as sym
from . import Functions_Base as _base


class Funcs1D(_base.FuncBase):

	_sympy_var = None

	def __init__(self, domain=lambda _: True):
		super().__init__(domain=domain, input_dim=1, output_dim=1)

	@property
	def sympy_var(self):
		if self._sympy_var is not None:
			return self._sympy_var
		return sym.Symbol('x', real=True)

	@sympy_var.setter
	def sympy_var(self, value):
		self._sympy_var = value

	def derivative_n(self, n):
		if n < 0:
			raise ValueError("n must be non-negative")
		result = self
		for _ in range(n):
			result = result.derivative()
		return result

	def numerical_integrate(self, a, b):
		from scipy.integrate import quad
		result, _ = quad(lambda x: float(self(x, ignore_domain=True)), a, b)
		return result

	def integrate(self, a=None, b=None):
		if a is None and b is None:
			return self.antiderivative()
		if a is not None and b is not None:
			try:
				F = self.antiderivative()
				return F(b, ignore_domain=True) - F(a, ignore_domain=True)
			except NotImplementedError:
				return self.numerical_integrate(a, b)
		raise ValueError(
			"integrate() takes either no positional arguments (indefinite) "
			"or exactly two (definite interval [a, b])"
		)




################
# ampl * exp(k * (x - shift))
class ExpFunc(Funcs1D):

	def __new__(cls, k, ampl=1, shift=0, domain=lambda _: True):
		if ampl == 0:
			return _base.ZeroFunc(domain=domain, input_dim=1)
		if k == 0:
			return _base.ConstFunc(ampl, domain=domain, input_dim=1)
		return object.__new__(cls)

	def __init__(self, k, ampl=1, shift=0, domain=lambda _: True):
		super().__init__(domain=domain)
		self.k = k
		self.ampl = ampl
		self.shift = shift
		self.parameters = {'k': k, 'ampl': ampl, 'shift': shift, 'domain': domain}

	def _eval(self, pos_arr):
		return self.ampl * np.exp(self.k * (pos_arr - self.shift))

	def __add__(self, other):
		if isinstance(other, ExpFunc) and self.k == other.k:
			domain = _base.combine_domains(self, other)
			s_mean = (self.shift + other.shift) / 2
			new_ampl = (self.ampl  * np.exp(self.k  * (s_mean - self.shift))
					  + other.ampl * np.exp(other.k * (s_mean - other.shift)))
			return ExpFunc(self.k, ampl=new_ampl, shift=s_mean, domain=domain)
		if isinstance(other, ExpFunc):
			return SumOfExps([self.ampl, other.ampl], [self.k, other.k],
							 shifts=[self.shift, other.shift],
							 domain=_base.combine_domains(self, other))
		if isinstance(other, SumOfExps):
			return other + self
		return super().__add__(other)

	def __mul__(self, other):
		val = _base.scalar_factor(other)
		if val is not None:
			return self.clone(ampl=val * self.ampl)
		return super().__mul__(other)

	def derivative(self):
		return self.clone(ampl=self.k * self.ampl)

	def antiderivative(self):
		return self.clone(ampl=self.ampl / self.k)

	def simplify(self, deep=False, **kwargs):
		return self

	def sympy_output(self):
		x = self.sympy_var
		return sym.sympify(self.ampl) * sym.exp(sym.sympify(self.k) * (x - sym.sympify(self.shift)))


################
# ampl * x^power
class PowFunc(Funcs1D):

	def __new__(cls, power, ampl=1, domain=lambda _: True):
		if ampl == 0:
			return _base.ZeroFunc(domain=domain, input_dim=1)
		return object.__new__(cls)

	def __init__(self, power, ampl=1, domain=lambda _: True):
		super().__init__(domain=domain)
		self.power = power
		self.ampl = ampl
		self.parameters = {'power': power, 'ampl': ampl, 'domain': domain}

	def _eval(self, pos_arr):
		return self.ampl * pos_arr ** self.power

	def __add__(self, other):
		if isinstance(other, PowFunc) and self.power == other.power:
			return PowFunc(self.power, ampl=self.ampl + other.ampl, domain=_base.combine_domains(self, other))
		return super().__add__(other)

	def __mul__(self, other):
		val = _base.scalar_factor(other)
		if val is not None:
			return self.clone(ampl=val * self.ampl)
		if isinstance(other, PowFunc):
			return PowFunc(self.power + other.power, ampl=self.ampl * other.ampl, domain=_base.combine_domains(self, other))
		return super().__mul__(other)

	def derivative(self):
		if self.power == 0:
			return _base.ZeroFunc(domain=self.domain, input_dim=1)
		return self.clone(power=self.power - 1, ampl=self.power * self.ampl)

	def antiderivative(self):
		if self.power == -1:
			raise NotImplementedError(
				"PowFunc with power=-1 (1/x) has no PowFunc antiderivative; "
				"call integrate(a, b) for a definite integral"
			)
		return self.clone(power=self.power + 1, ampl=self.ampl / (self.power + 1))

	def sympy_output(self):
		x = self.sympy_var
		return sym.sympify(self.ampl) * x ** self.power


################
# coeffs[i] is the coefficient of x^i (ascending powers)
class PolyFunc(Funcs1D):

	def __new__(cls, coeffs, domain=lambda _: True):
		if np.all(np.asarray(coeffs) == 0):
			return _base.ZeroFunc(domain=domain, input_dim=1)
		return object.__new__(cls)

	def __init__(self, coeffs, domain=lambda _: True):
		super().__init__(domain=domain)
		coeffs = np.asarray(coeffs)
		dtype = np.result_type(float, coeffs.dtype)
		coeffs = coeffs.astype(dtype)
		nonzero_idx = np.flatnonzero(coeffs)
		self.coeffs = coeffs[:nonzero_idx[-1] + 1]
		self.parameters = {'coeffs': self.coeffs, 'domain': domain}

	def _eval(self, pos_arr):
		return np.polynomial.polynomial.polyval(pos_arr, self.coeffs)

	def __add__(self, other):
		if isinstance(other, PolyFunc):
			n = max(len(self.coeffs), len(other.coeffs))
			c1 = np.pad(self.coeffs, (0, n - len(self.coeffs)))
			c2 = np.pad(other.coeffs, (0, n - len(other.coeffs)))
			return PolyFunc(c1 + c2, domain=_base.combine_domains(self, other))
		if isinstance(other, _base.ConstFunc) and other.output_dim == 1:
			dtype = np.result_type(self.coeffs.dtype, np.array(other.const).dtype)
			new_coeffs = self.coeffs.astype(dtype)
			new_coeffs[0] += other.const
			return PolyFunc(new_coeffs, domain=_base.combine_domains(self, other))
		if (isinstance(other, PowFunc) and np.isscalar(other.ampl)
				and other.power >= 0 and int(other.power) == other.power):
			p = int(other.power)
			n = max(len(self.coeffs), p + 1)
			dtype = np.result_type(self.coeffs.dtype, np.array(other.ampl).dtype)
			new_coeffs = np.pad(self.coeffs.astype(dtype), (0, n - len(self.coeffs)))
			new_coeffs[p] += other.ampl
			return PolyFunc(new_coeffs, domain=_base.combine_domains(self, other))
		return super().__add__(other)

	def __mul__(self, other):
		val = _base.scalar_factor(other)
		if val is not None:
			return self.clone(coeffs=val * self.coeffs)
		if isinstance(other, PolyFunc):
			return PolyFunc(np.convolve(self.coeffs, other.coeffs), domain=_base.combine_domains(self, other))
		if isinstance(other, PowFunc) and other.power >= 0 and int(other.power) == other.power:
			p = int(other.power)
			new_coeffs = np.concatenate([np.zeros(p), other.ampl * self.coeffs])
			return PolyFunc(new_coeffs, domain=_base.combine_domains(self, other))
		return super().__mul__(other)

	def derivative(self):
		if len(self.coeffs) <= 1:
			return _base.ZeroFunc(domain=self.domain, input_dim=1)
		new_coeffs = self.coeffs[1:] * np.arange(1, len(self.coeffs))
		return PolyFunc(new_coeffs, domain=self.domain)

	def antiderivative(self):
		anti = np.concatenate([[0.0], self.coeffs / np.arange(1, len(self.coeffs) + 1)])
		return PolyFunc(anti, domain=self.domain)

	def sympy_output(self):
		x = self.sympy_var
		return sum(sym.sympify(float(c)) * x**i for i, c in enumerate(self.coeffs) if c != 0)




################
# A class to represent a sum of exponentials
class SumOfExps(Funcs1D):

	def __new__(cls, coeffs, exponents, shifts=None, domain=lambda _: True):
		coeffs_arr = np.asarray(coeffs)
		if np.all(coeffs_arr == 0):
			return _base.ZeroFunc(domain=domain, input_dim=1)
		if len(coeffs_arr) == 1:
			shift = 0 if shifts is None else np.asarray(shifts)[0]
			return ExpFunc(exponents[0], ampl=coeffs[0], shift=shift, domain=domain)
		return object.__new__(cls)

	def __init__(self, coeffs, exponents, shifts=None, domain=lambda _: True):
		if len(coeffs) != len(exponents):
			raise ValueError("Length of coeffs and exponents must match.")
		if shifts is not None:
			if len(coeffs) != len(shifts):
				raise ValueError("Length of coeffs and shifts must match.")

		if shifts is None:
			shifts = np.zeros(len(exponents))

		super().__init__(domain=domain)
		self.coeffs = np.array(coeffs)
		self.exponents = np.array(exponents)
		self.shifts = np.array(shifts)
		self.parameters = {'coeffs': self.coeffs, 'exponents': self.exponents,
						   'shifts': self.shifts, 'domain': domain}

	def _eval(self, pos_arr):
		exp_args = self.exponents[:, None] * (pos_arr[None, :] - self.shifts[:, None])
		return self.coeffs @ np.exp(exp_args)

	# Returns the integral of this function mod squared over the interval [low,high]
	def norm(self, low, high):
		e1 = self.exponents[:, None]          # (N, 1)
		e2 = self.exponents[None, :].conj()   # (1, N)
		s1 = self.shifts[:, None]             # (N, 1)
		s2 = self.shifts[None, :].conj()      # (1, N)

		delta = (e1 + e2) / 2
		near_zero = (np.abs(delta * (2*high - s1 - s2)) < 1e-10) & \
		            (np.abs(delta * (2*low  - s1 - s2)) < 1e-10)

		exp_diff = (e1 - e2) / 2
		val_zero = np.exp(-exp_diff * (s1 - s2)) * (high - low)

		ep_high = np.exp(e1 * (high - s1)) * np.exp(e2 * (high - s2))
		ep_low  = np.exp(e1 * (low  - s1)) * np.exp(e2 * (low  - s2))
		safe_sum = np.where(near_zero, 1.0, e1 + e2)
		val_normal = (ep_high - ep_low) / safe_sum

		exp_integrals = np.where(near_zero, val_zero, val_normal)
		coeffs_prods = self.coeffs[:, None] * self.coeffs[None, :].conj()
		return np.sum((coeffs_prods * exp_integrals).real)

	def __add__(self, other):
		if isinstance(other, SumOfExps):
			o_coeffs, o_exps, o_shifts, o_domain = other.coeffs, other.exponents, other.shifts, other.domain
		elif isinstance(other, ExpFunc):
			o_coeffs, o_exps, o_shifts, o_domain = [other.ampl], [other.k], [other.shift], other.domain
		elif isinstance(other, _base.ConstFunc) and other.output_dim == 1:
			o_coeffs, o_exps, o_shifts, o_domain = [other.const], [0], [0], other.domain
		else:
			val = _base.as_scalar(other)
			if val is None:
				return super().__add__(other)
			if val == 0:
				return self.copy()
			o_coeffs, o_exps, o_shifts, o_domain = [val], [0], [0], (lambda _: True)
		coeffs = np.concatenate((self.coeffs, o_coeffs))
		exponents = np.concatenate((self.exponents, o_exps))
		shifts = np.concatenate((self.shifts, o_shifts))
		return SumOfExps(coeffs=coeffs, exponents=exponents, shifts=shifts,
		                 domain=_base.combine_domains(self.domain, o_domain))

	def __mul__(self, other):
		if isinstance(other, SumOfExps):
			exp_tot      = self.exponents[:, None] + other.exponents[None, :]
			coeff_tot    = self.coeffs[:, None]    * other.coeffs[None, :]
			weight_shift = (self.exponents[:, None]  * self.shifts[:, None]
			              + other.exponents[None, :] * other.shifts[None, :])
			tiny      = np.abs(weight_shift) > np.abs(exp_tot) * 1e10
			safe_exp  = np.where(tiny, 1.0, exp_tot)
			shift_tot = np.where(tiny, 0.0, weight_shift / safe_exp)
			coeff_tot = np.where(tiny, coeff_tot * np.exp(-weight_shift), coeff_tot)
			domain = _base.combine_domains(self, other)
			return SumOfExps(coeffs=coeff_tot.ravel(), exponents=exp_tot.ravel(),
			                 shifts=shift_tot.ravel(), domain=domain).simplify()
		val = _base.scalar_factor(other)
		if val is not None:
			return self.clone(coeffs=val * self.coeffs)
		return super().__mul__(other)

	def _check_approx(self, result, t_samples, atol, rtol, label):
		orig = self(t_samples)
		appr = result(t_samples)
		abs_err = np.abs(appr - orig)
		scale   = np.where(np.abs(orig) > 0, np.abs(orig), 1.0)
		max_abs = abs_err.max()
		max_rel = (abs_err / scale).max()
		n_out   = len(result.coeffs) if isinstance(result, SumOfExps) else 0
		print(f"{label}: {len(self.coeffs)} → {n_out} terms, "
		      f"max abs err = {max_abs:.2e}, max rel err = {max_rel:.2e}")
		if atol is not None and max_abs > atol:
			raise ValueError(f"{label}: max abs error {max_abs:.2e} exceeds atol={atol}")
		if rtol is not None and max_rel > rtol:
			raise ValueError(f"{label}: max rel error {max_rel:.2e} exceeds rtol={rtol}")

	def simplify(self, deep=False, exp_threshold=0.0, coeff_threshold=0.0, t_samples=None, atol=None, rtol=None):
		"""Merge exponents, drop small terms, and optionally verify accuracy.

		deep            : accepted for API consistency with SumOfFuncs/ProdOfFuncs, which call
		                  f.simplify(deep=True) on sub-expressions. Has no effect here — SumOfExps
		                  has no sub-expressions to recurse into.
		exp_threshold   : merge exponents within this absolute tolerance (0 = exact match only).
		                  Single-grid scheme with cell size exp_threshold/2; pairs that straddle a
		                  cell boundary may not be merged.
		coeff_threshold : drop terms with |c| < coeff_threshold * max|c| after merging (0 = keep all).
		t_samples       : optional sample points for an error report / tolerance check.
		atol, rtol      : raise ValueError if max abs/rel error exceeds these (requires t_samples).
		"""
		if (atol is not None or rtol is not None) and t_samples is None:
			raise ValueError("t_samples is required when atol or rtol is specified")
		if len(self.coeffs) == 0:
			return self.copy()

		re_arr = np.real(self.exponents)
		im_arr = np.imag(self.exponents)

		if exp_threshold > 0:
			keys_re = np.floor(re_arr * 2 / exp_threshold).astype(np.int64)
			keys_im = np.floor(im_arr * 2 / exp_threshold).astype(np.int64)
		else:
			keys_re, keys_im = re_arr, im_arr

		order    = np.lexsort((keys_im, keys_re))
		is_new   = np.ones(len(order), dtype=bool)
		is_new[1:] = (keys_re[order[1:]] != keys_re[order[:-1]]) | (keys_im[order[1:]] != keys_im[order[:-1]])
		n_groups = int(is_new.sum())
		group_id = np.empty(len(order), dtype=np.intp)
		group_id[order] = np.cumsum(is_new) - 1

		exp_sum  = np.zeros(n_groups, dtype=complex)
		np.add.at(exp_sum, group_id, self.exponents)
		new_exponents = exp_sum / np.bincount(group_id)

		shift_sum = np.zeros(n_groups, dtype=complex)
		np.add.at(shift_sum, group_id, self.shifts)
		new_shifts = shift_sum / np.bincount(group_id)

		new_coeffs = np.zeros(n_groups, dtype=complex)
		np.add.at(new_coeffs, group_id,
		          self.coeffs * np.exp(-new_exponents[group_id] * (self.shifts - new_shifts[group_id])))

		result = SumOfExps(coeffs=new_coeffs, exponents=new_exponents,
		                   shifts=new_shifts, domain=self.domain)

		if coeff_threshold > 0 and isinstance(result, SumOfExps):
			abs_c  = np.abs(result.coeffs)
			mask   = abs_c > coeff_threshold * abs_c.max()
			result = SumOfExps(coeffs=result.coeffs[mask], exponents=result.exponents[mask],
			                   shifts=result.shifts[mask], domain=result.domain)

		if t_samples is not None:
			self._check_approx(result, t_samples, atol, rtol,
			                   label=f"simplify(exp_threshold={exp_threshold}, coeff_threshold={coeff_threshold})")
		return result

	def derivative(self):
		return self.clone(coeffs=self.exponents * self.coeffs)

	def antiderivative(self):
		if np.any(self.exponents == 0):
			raise NotImplementedError(
				"SumOfExps with k=0 terms has no SumOfExps antiderivative; "
				"call integrate(a, b) for a definite integral"
			)
		return self.clone(coeffs=self.coeffs / self.exponents)

	def integrate(self, a=None, b=None):
		if a is None and b is None:
			return self.antiderivative()
		if a is not None and b is not None:
			result = 0.0
			for c, k, s in zip(self.coeffs, self.exponents, self.shifts):
				if k == 0:
					result = result + c * (b - a)
				else:
					result = result + c / k * (np.exp(k * (b - s)) - np.exp(k * (a - s)))
			return result
		raise ValueError(
			"integrate() takes either no positional arguments (indefinite) "
			"or exactly two (definite interval [a, b])"
		)

	def sympy_output(self):
		x = self.sympy_var
		return sum(c * sym.exp(e * (x - s)) for c, e, s in zip(self.coeffs, self.exponents, self.shifts))
