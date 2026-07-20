from yunmeng.solutions.standards import ILinkableModel
from yunmeng.solutions.commons import BaseModel

import importlib
import pkgutil
import inspect
from pathlib import Path

# register all models here.
ym_models = {}
for _, module_name, _ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if module_name.startswith("_"):
        continue
    if module_name == "standards" or module_name == "commons":
        continue

    module = importlib.import_module(f".{module_name}", __package__)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, ILinkableModel)
            and obj is not ILinkableModel
            and obj is not BaseModel
        ):
            if name in ym_models:
                raise ValueError(f"Duplicated model: {name}.")
            globals()[name] = obj
            ym_models[name] = obj

# import all the fvm solvers here.
__all__ = [
    name
    for name, obj in globals().items()
    if inspect.isclass(obj)
    and issubclass(obj, ILinkableModel)
    and obj is not ILinkableModel
]
