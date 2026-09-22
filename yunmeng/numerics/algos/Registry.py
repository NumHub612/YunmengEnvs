# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Unified components registry.
"""

from yunmeng.interfaces.capabilities import IEstimator, IScheduler
from yunmeng.interfaces.solution import ILinkableModel, IModelCallback
from yunmeng.interfaces.solver import (
    ISolver,
    IOperator,
    IInitCondition,
    IBoundaryCondition,
    ISolverCallback,
)

_KINDS = (
    # capabilities
    "estimator",  # IEstimator
    "scheduler",  # IScheduler
    # solution
    "model",  # ILinkableModel
    "event",  # IModelCallback
    # solver
    "solver",  # ISolver
    "operator",  # IOperator
    "init",  # IInitCondition
    "boundary",  # IBoundaryCondition
    "callback",  # ISolverCallback
)

_KINDS_MAP = {
    "estimator": IEstimator,
    "scheduler": IScheduler,
    "model": ILinkableModel,
    "event": IModelCallback,
    "solver": ISolver,
    "operator": IOperator,
    "init": IInitCondition,
    "boundary": IBoundaryCondition,
    "callback": ISolverCallback,
}

# name -> {"cls": type, "kind": str}; name globally unique
_REGISTRY: dict[str, dict] = {}


def ym_register(kind: str, cls: type = None):
    """Registry algorithm class (mainly as decorator).

    @ym_register("model")
    class FooModel(ILinkableModel):

        @classmethod
        def get_name(cls) -> str:
            return "foo"
        ...

    # 直接调用形式(动态注册场景)
    ym_register("model", FooModel)
    """
    # kind checking
    if kind not in _KINDS:
        raise KeyError(f"Unknown component kind '{kind}' (known: {list(_KINDS)}).")

    # decorator form
    if cls is None:
        return lambda c: ym_register(kind, c)

    # class checking
    target_type = _KINDS_MAP[kind]
    if not (isinstance(cls, type) and issubclass(cls, target_type)):
        raise TypeError(f"{cls} is not a subclass of {target_type} of kind '{kind}'")

    reg_name = cls.get_name() if hasattr(cls, "get_name") else None
    if not reg_name:
        raise ValueError(f"{cls.__name__} must implement get_name()")
    if reg_name in _REGISTRY and _REGISTRY[reg_name]["cls"] is not cls:
        raise KeyError(
            f"Class name '{reg_name}' has already been registered by "
            f"{_REGISTRY[reg_name]['cls'].__name__}({_REGISTRY[reg_name]['kind']}), "
            f"registry name must be globally unique"
        )

    _REGISTRY[reg_name] = {"cls": cls, "kind": kind}
    return cls


def get_class(name: str) -> type:
    """Get component class by name."""
    try:
        return _REGISTRY[name]["cls"]
    except KeyError:
        raise KeyError(
            f"No component named '{name}'; available: {sorted(_REGISTRY)}."
        ) from None


def kind_of(name: str) -> str:
    """Reverse lookup component's kind by name."""
    try:
        return _REGISTRY[name]["kind"]
    except KeyError:
        raise KeyError(f"No component named '{name}'.") from None


def create(name: str, **params) -> object:
    """Create component instance by name."""
    return get_class(name)(**params)


def available(kind: str = None) -> dict:
    """Query registered component. When kind=None, return the full table as {kind: [names]}."""
    if kind is not None:
        if kind not in _KINDS:
            raise KeyError(f"Unknown kind '{kind}' (known: {list(_KINDS)}).")
        return sorted(n for n, r in _REGISTRY.items() if r["kind"] == kind)
    table = {k: [] for k in _KINDS}
    for n, r in _REGISTRY.items():
        table[r["kind"]].append(n)
    return {k: sorted(v) for k, v in table.items()}
