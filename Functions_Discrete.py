import numpy as np
import sympy as sym
from abc import abstractmethod

from .Functions_Base import FuncBase, ZeroFunc, ConstFunc, gen_sum, gen_prod


class FuncsDiscrete(FuncBase):
    """Abstract base for functions on ℤ (or subsets defined by domain)."""

    _sympy_var = sym.Symbol('n', integer=True)

    def __init__(self, domain=lambda _: True):
        super().__init__(domain=domain, input_dim=0, output_dim=0)

    @abstractmethod
    def _eval(self, pos_arr):
        """pos_arr: (N,) integer array. Return (N,) array."""
        pass

    @abstractmethod
    def derivative(self, direction='forward'):
        """Return a new FuncsDiscrete for the forward (or backward) difference."""
        pass

    @abstractmethod
    def sympy_output(self):
        pass

    @abstractmethod
    def copy(self):
        pass


class ExpSeq(FuncsDiscrete):
    """Geometric sequence: ampl * base**n."""

    def __new__(cls, base, ampl=1, domain=lambda _: True):
        if ampl == 0:
            return ZeroFunc(domain=domain, input_dim=0)
        if base == 0:
            raise ValueError("base=0 is degenerate (0**0 undefined); use TabFunc instead")
        if base == 1:
            return ConstFunc(ampl, domain=domain, input_dim=0)
        return object.__new__(cls)

    def __init__(self, base, ampl=1, domain=lambda _: True):
        super().__init__(domain=domain)
        self.base = base
        self.ampl = ampl

    def _eval(self, pos_arr):
        return self.ampl * (self.base ** pos_arr)

    def derivative(self, direction='forward'):
        if direction == 'forward':
            return ExpSeq(self.base, ampl=self.ampl * (self.base - 1), domain=self.domain)
        if direction == 'backward':
            return ExpSeq(self.base, ampl=self.ampl * (1 - 1 / self.base), domain=self.domain)
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

    def __add__(self, other):
        if isinstance(other, ExpSeq) and self.base == other.base:
            return ExpSeq(self.base, ampl=self.ampl + other.ampl,
                          domain=lambda p: self.domain(p) and other.domain(p))
        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if np.isscalar(other):
            return ExpSeq(self.base, ampl=other * self.ampl, domain=self.domain)
        if isinstance(other, ExpSeq):
            return ExpSeq(self.base * other.base, ampl=self.ampl * other.ampl,
                          domain=lambda p: self.domain(p) and other.domain(p))
        return super().__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def sympy_output(self):
        n = self._sympy_var
        return sym.sympify(self.ampl) * sym.sympify(self.base) ** n

    def copy(self):
        return ExpSeq(self.base, ampl=self.ampl, domain=self.domain)


class PolySeq(FuncsDiscrete):
    """Polynomial in n: sum of coeffs[k] * n**k (ascending powers)."""

    def __new__(cls, coeffs, domain=lambda _: True):
        if np.all(np.asarray(coeffs) == 0):
            return ZeroFunc(domain=domain, input_dim=0)
        return object.__new__(cls)

    def __init__(self, coeffs, domain=lambda _: True):
        super().__init__(domain=domain)
        arr = np.asarray(coeffs, dtype=float)
        nz = np.flatnonzero(arr)
        self.coeffs = arr[:nz[-1] + 1]

    def _eval(self, pos_arr):
        return np.polynomial.polynomial.polyval(pos_arr.astype(float), self.coeffs)

    def derivative(self, direction='forward'):
        if direction not in ('forward', 'backward'):
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
        n = self._sympy_var
        expr = self.sympy_output()
        shift = +1 if direction == 'forward' else -1
        diff_expr = sym.expand(expr.subs(n, n + shift) - expr)
        if diff_expr == 0:
            return ZeroFunc(domain=self.domain, input_dim=0)
        poly = sym.Poly(diff_expr, n)
        coeffs = [float(c) for c in reversed(poly.all_coeffs())]
        return PolySeq(coeffs, domain=self.domain)

    def __add__(self, other):
        if isinstance(other, PolySeq):
            size = max(len(self.coeffs), len(other.coeffs))
            c1 = np.pad(self.coeffs, (0, size - len(self.coeffs)))
            c2 = np.pad(other.coeffs, (0, size - len(other.coeffs)))
            return PolySeq(c1 + c2, domain=lambda p: self.domain(p) and other.domain(p))
        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if np.isscalar(other):
            return PolySeq(other * self.coeffs, domain=self.domain)
        if isinstance(other, PolySeq):
            return PolySeq(np.convolve(self.coeffs, other.coeffs),
                           domain=lambda p: self.domain(p) and other.domain(p))
        return super().__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def sympy_output(self):
        n = self._sympy_var
        return gen_sum([sym.sympify(float(c)) * n**k
                        for k, c in enumerate(self.coeffs) if c != 0])

    def copy(self):
        return PolySeq(self.coeffs.copy(), domain=self.domain)


class TabFunc(FuncsDiscrete):
    """Lookup-table sequence: dict {n: value} or array indexed from 0."""

    def __init__(self, data, default=0, domain=lambda _: True):
        super().__init__(domain=domain)
        if isinstance(data, dict):
            self._table = {int(k): float(v) for k, v in data.items()}
        else:
            arr = np.asarray(data, dtype=float)
            self._table = {i: float(v) for i, v in enumerate(arr) if v != 0}
        self.default = float(default)

    def _eval(self, pos_arr):
        return np.array([self._table.get(int(n), self.default) for n in pos_arr])

    def derivative(self, direction='forward'):
        if direction == 'forward':
            keys = set(self._table) | {k - 1 for k in self._table}
            new = {n: self._table.get(n + 1, self.default) - self._table.get(n, self.default)
                   for n in keys}
        elif direction == 'backward':
            keys = set(self._table) | {k + 1 for k in self._table}
            new = {n: self._table.get(n, self.default) - self._table.get(n - 1, self.default)
                   for n in keys}
        else:
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
        new = {k: v for k, v in new.items() if v != 0}
        new_default = self.default - self.default
        return TabFunc(new, default=new_default, domain=self.domain)

    def __add__(self, other):
        if isinstance(other, TabFunc):
            keys = set(self._table) | set(other._table)
            new = {k: self._table.get(k, self.default) + other._table.get(k, other.default)
                   for k in keys}
            return TabFunc(new, default=self.default + other.default,
                           domain=lambda p: self.domain(p) and other.domain(p))
        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if np.isscalar(other):
            return TabFunc({k: other * v for k, v in self._table.items()},
                           default=other * self.default, domain=self.domain)
        return super().__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def sympy_output(self):
        n = self._sympy_var
        pieces = [(sym.sympify(v), sym.Eq(n, sym.Integer(k)))
                  for k, v in sorted(self._table.items())]
        pieces.append((sym.sympify(self.default), True))
        return sym.Piecewise(*pieces)

    def copy(self):
        return TabFunc(dict(self._table), default=self.default, domain=self.domain)
