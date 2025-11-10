from core.solvers.fvm.operators import fvm_operators
from core.solvers.interfaces.ISolver import ISolver
from core.solvers.commons import BaseSolver

import importlib
import pkgutil
import inspect
from pathlib import Path

# register all the fvm solvers here.
fvm_solvers = {}
for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if module_name.startswith("_"):
        continue

    module = importlib.import_module(f".{module_name}", __package__)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, ISolver) and obj is not ISolver and obj is not BaseSolver:
            if name in fvm_solvers:
                raise ValueError(f"Solver name {name} already exists.")
            globals()[name] = obj
            fvm_solvers[obj.get_name()] = obj

# import all the fvm solvers here.
__all__ = [
    name
    for name, obj in globals().items()
    if inspect.isclass(obj) and issubclass(obj, ISolver) and obj is not ISolver
]
