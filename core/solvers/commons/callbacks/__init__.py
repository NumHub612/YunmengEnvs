from core.solvers.commons.callbacks.ImageRender import *
from core.solvers.commons.callbacks.PerformanceMonitor import *


# register all solver callbacks here
callback_handlers = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, ISolverCallback):
        if name == "ISolverCallback":
            continue
        if name in callback_handlers:
            raise ValueError(f"Duplicated solver callback: {name}.")
        callback_handlers[obj.get_name()] = obj
