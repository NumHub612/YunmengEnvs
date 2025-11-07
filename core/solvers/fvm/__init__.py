from core.solvers.interfaces.ISolver import ISolver
from core.solvers.fvm.operators import fvm_operators
from core.solvers.fvm.Burgers2D import *


# register all the fvm solvers here.
fvm_solvers = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, ISolver):
        if name == "ISolver":
            continue
        if name in fvm_solvers:
            raise ValueError(f"Duplicated solver: {name}.")
        fvm_solvers[obj.get_name()] = obj
