from yunmeng.solvers.fdm.operators.curls import *
from yunmeng.solvers.fdm.operators.divs import *
from yunmeng.solvers.fdm.operators.laplacians import *
from yunmeng.solvers.fdm.operators.grads import *
from yunmeng.solvers.fdm.operators.d2dt2s import *
from yunmeng.solvers.fdm.operators.ddts import *
from yunmeng.solvers.fdm.operators.srcs import *


# register all the fdm operators
fdm_operators = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, IOperator):
        if name == "IOperator":
            continue
        if name in fdm_operators:
            raise ValueError(f"Duplicated fdm operator: {name}.")
        fdm_operators[name] = obj
