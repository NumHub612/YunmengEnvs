from core.solvers.commons.boundaries.custom import *
from core.solvers.commons.boundaries.constant import *
from core.solvers.commons.boundaries.dirichlets import *


# register all the boundary conditions.
boundary_conditions = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, IBoundaryCondition):
        if name == "IBoundaryCondition":
            continue
        if name in boundary_conditions:
            raise ValueError(f"Duplicated boundary condition: {name}.")
        boundary_conditions[obj.get_name()] = obj
