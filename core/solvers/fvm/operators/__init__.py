from core.solvers.fvm.operators.curls import *
from core.solvers.fvm.operators.divs import *
from core.solvers.fvm.operators.laplacians import *
from core.solvers.fvm.operators.grads import *
from core.solvers.fvm.operators.d2dt2s import *
from core.solvers.fvm.operators.ddts import *
from core.solvers.fvm.operators.srcs import *


# register all the fvm operators
fvm_operators = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, IOperator):
        if name == "IOperator":
            continue
        if name in fvm_operators:
            raise ValueError(f"Duplicated fvm operator: {name}.")
        fvm_operators[name] = obj
