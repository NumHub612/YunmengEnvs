from core.solutions.standards import ILinkableComponent
from core.solutions.HydroModels import *


# register all models here.
ym_models = {}
for name, obj in list(locals().items()):
    if isinstance(obj, type) and issubclass(obj, ILinkableComponent):
        if name == "ILinkableComponent":
            continue
        if name in ym_models:
            raise ValueError(f"Duplicated model name: {name}.")
        ym_models[name] = obj
