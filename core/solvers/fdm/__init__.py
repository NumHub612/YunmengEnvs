from core.solvers.fdm.Burgers1D import *


# register all the fdm solvers here.
fdm_solvers = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, ISolver):
        if name == "ISolver":
            continue
        if name in fdm_solvers:
            raise ValueError(f"Duplicated solver: {name}.")
        fdm_solvers[obj.get_name()] = obj
