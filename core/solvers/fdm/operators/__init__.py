from core.solvers.fdm.operators.curls import *
from core.solvers.fdm.operators.divs import *
from core.solvers.fdm.operators.laplacians import *
from core.solvers.fdm.operators.grads import *
from core.solvers.fdm.operators.d2dt2s import *
from core.solvers.fdm.operators.ddts import *
from core.solvers.fdm.operators.srcs import *


# register all the fdm operators
fdm_operators = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, IOperator):
        if name == "IOperator":
            continue
        if name in fdm_operators:
            raise ValueError(f"Duplicated fdm operator: {name}.")
        fdm_operators[name] = obj
