"""
FDM solvers are mainly developed on grids for the purpose of studying the
numerical discretization of PDEs.
"""

from yunmeng.solvers.fdm.operators import fdm_operators
from yunmeng.solvers.interfaces.ISolver import ISolver
from yunmeng.solvers.commons import BaseSolver

import importlib
import pkgutil
import inspect
from pathlib import Path

# register all the fdm solvers here.
fdm_solvers = {}
for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if module_name.startswith("_"):
        continue

    module = importlib.import_module(f".{module_name}", __package__)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ISolver) and obj is not ISolver and obj is not BaseSolver:
            globals()[name] = obj
            fdm_solvers[obj.get_name()] = obj

# import all the fdm solvers here.
__all__ = [
    name
    for name, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, ISolver) and obj is not ISolver
]
