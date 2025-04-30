from core.solvers.interfaces.ISolver import ISolver


# register all the fvm solvers here.
fvm_solvers = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, ISolver):
        if name == "ISolver":
            continue
        if name in fvm_solvers:
            raise ValueError(f"Duplicated solver: {name}.")
        fvm_solvers[obj.get_name()] = obj
