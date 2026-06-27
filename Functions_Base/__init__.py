"""Core of the Functions library: the ``FuncBase`` ABC and the primitive/combinator classes.

Formerly a single ``Functions_Base.py`` module; split into submodules for size while keeping the
import path identical (``from . import Functions_Base as _base``, ``_base.ZeroFunc``, …). The
submodules are mutually recursive at call time (the ABC's operators construct the leaves and
combinators, which subclass the ABC); the cycle is broken with module-bottom imports — see the
notes at the end of ``func_base.py`` and ``leaves.py``.

  - func_base.py   : FuncBase ABC + helpers (as_scalar, scalar_factor, combine_domains,
                     _check_dim_compat, _fuse_pairwise, _SENTINEL)
  - leaves.py      : ZeroFunc, ConstFunc, CoordPow, TrigCoord
  - combinators.py : SumOfFuncs, ProdOfFuncs, RatioOfFuncs, ComposedFunc, VecFunc
"""

from .func_base import (
	FuncBase, as_scalar, scalar_factor, combine_domains, _check_dim_compat, _fuse_pairwise, _SENTINEL,
)
from .leaves import ZeroFunc, ConstFunc, CoordPow, TrigCoord
from .combinators import SumOfFuncs, ProdOfFuncs, RatioOfFuncs, ComposedFunc, VecFunc

__all__ = [
	'FuncBase', 'ZeroFunc', 'ConstFunc',
	'SumOfFuncs', 'ProdOfFuncs', 'RatioOfFuncs', 'ComposedFunc', 'VecFunc',
	'CoordPow', 'TrigCoord',
	'as_scalar', 'scalar_factor', 'combine_domains', '_check_dim_compat', '_fuse_pairwise', '_SENTINEL',
]
