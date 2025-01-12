from core.solvers.commons.inits.uniform import *
from core.solvers.commons.inits.custom import *
from core.solvers.commons.inits.hotstart import *


# register all the init methods
init_methods = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, IInitCondition):
        if name == "IInitCondition":
            continue
        if name in init_methods:
            raise ValueError(f"Duplicated init method: {name}.")
        init_methods[obj.get_name()] = obj
