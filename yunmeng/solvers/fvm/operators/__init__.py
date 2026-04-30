from yunmeng.solvers.fvm.operators.curls import *
from yunmeng.solvers.fvm.operators.divs import *
from yunmeng.solvers.fvm.operators.laplacians import *
from yunmeng.solvers.fvm.operators.grads import *
from yunmeng.solvers.fvm.operators.d2dt2s import *
from yunmeng.solvers.fvm.operators.ddts import *
from yunmeng.solvers.fvm.operators.srcs import *
from yunmeng.solvers.fvm.operators.funcs import *


# register all the fvm operators
fvm_operators = {}
for _, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, IOperator):
        name = obj.get_name()
        if name == None:
            continue
        if name in fvm_operators:
            raise ValueError(f"Duplicated fvm operator: {name}.")
        fvm_operators[name] = obj
