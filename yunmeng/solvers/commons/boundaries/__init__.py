from yunmeng.solvers.interfaces import *
from yunmeng.solvers.commons.boundaries.value import *
from yunmeng.solvers.commons.boundaries.flux import *

# register all the boundary conditions.
boundary_conditions = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, IBoundaryCondition):
        if name == "IBoundaryCondition":
            continue
        if name == "BaseBoundary":
            continue
        if name in boundary_conditions:
            raise ValueError(f"Duplicated boundary condition: {name}.")
        boundary_conditions[obj.get_name()] = obj
