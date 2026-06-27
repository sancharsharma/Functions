"""Make `import Functions` resolve when running the tests in place.

The package directory is `.../MyPythonLib/Functions`, imported as the package `Functions`
(see `package-dir` in pyproject.toml). When the package isn't installed, put the parent
`MyPythonLib` directory on sys.path so `import Functions` works from a bare checkout.
"""
import sys
from pathlib import Path

_MYPYTHONLIB = Path(__file__).resolve().parents[2]  # .../MyPythonLib
if str(_MYPYTHONLIB) not in sys.path:
	sys.path.insert(0, str(_MYPYTHONLIB))
