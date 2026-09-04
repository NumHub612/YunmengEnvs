# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solver-layer protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields as dc_fields
from typing import Any, Protocol

from yunmeng.interfaces.support.field import FieldMeta, IField
from yunmeng.interfaces.solver.IBoundaryCondition import (
    IBoundaryProvider,
    IBoundaryCondition,
)
from yunmeng.interfaces.solver.IInitCondition import IInitialCondition
from yunmeng.interfaces.solver.IEquation import IEquation
from yunmeng.interfaces.solver.ISolverCallback import ISolverCallback
from yunmeng.interfaces.types import MeshDimension, RunMode

# ---------------------------------------------------
# region Solver meta
# ---------------------------------------------------


@dataclass
class SolverMeta:
    """Solver meta description."""

    description: str = ""  # Brief description about the solver.
    kind: str = "unknown"  # open kind: "fvm"/"fdm"/"fem"/...
    equation: str = ""  # The equation to be solved, e.g. Swe2D.
    equation_expr: str = ""  # The mathematical expression.
    dimension: MeshDimension = MeshDimension.NONE
    default_ics: dict[str, IInitialCondition] = None
    default_bcs: dict[str, IBoundaryCondition] = None
    fields: dict[str, FieldMeta] = None  # The available fields.


@dataclass
class SolverStatus:
    """Current solver status."""

    current_time: float = None  # Current physical time.
    end_time: float = None  # End time.
    time_step: float = None  # Current time step.
    residual: float = None  # Current residual.
    step_time: float = None  # Current step time.
    total_time: float = None  # Total elapsed time.
    steps: int = None  # Current simulation steps.
    iters: int = None  # Current iterations.
    finished: bool = False  # Whether finished.
    errors: str = None  # Error message.
    extras: Any = None  # Any extra information used.


@dataclass
class SolverConfig(ABC):
    """Base class for solver configurations."""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SolverConfig":
        """Build from a plain dict; unknown keys silently ignored."""
        keys = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in d.items() if k in keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        result = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            item = getattr(val, "item", None)
            if callable(item) and not isinstance(val, (str, bytes, dict, list, tuple)):
                try:
                    val = item()
                except (ValueError, TypeError):
                    pass
            result[f.name] = val
        return result

    @classmethod
    @abstractmethod
    def get_solver_name(cls) -> str:
        """The solver class name this config belongs to."""
        ...


# ---------------------------------------------------
# region ISolver
# ---------------------------------------------------


class ISolver(Protocol):
    """Inference-time solver interface.

    Lifecycle:
        1. Construction: __init__(id, mesh, operators, config)
        2. Assembly: add_*, remove_*, clear_* (repeatable)
        3. Initialize: initialize(...) (once per run)
        4. Runtime: step(dt) (repeatedly)
    """

    # -- class metadata -----------------------------

    @classmethod
    def get_meta(cls) -> SolverMeta: ...

    @classmethod
    def get_name(cls) -> str:
        """The unique solver class name."""
        ...

    @classmethod
    def get_config_class(cls) -> type[SolverConfig]: ...

    # -- properties ---------------------------------

    @property
    def id(self) -> str: ...

    @property
    def status(self) -> SolverStatus: ...

    @property
    def config(self) -> SolverConfig: ...

    @property
    def mode(self) -> RunMode: ...

    # -- mode switchin ------------------------------

    def train(self):
        """Set TRAIN and propagate:
        solver -> DataHub -> IModeSwitchable operators (isinstance-checked).
        BaseSolver provides the default propagation;
        overrides must call super()."""
        ...

    def eval(self):
        """Set EVAL and propagate (default; backward compatible)."""
        ...

    # -- assembly -----------------------------------

    def set_problems(self, equations: list["IEquation"]):
        """Set the PDE problems to be solved."""
        ...

    def get_solution(self, field: str) -> IField:
        """Get the solution field by name."""
        ...

    def add_ic(self, ic: IInitialCondition):
        """Add an initial condition."""
        ...

    def clear_ics(self, field: str = None):
        """Clear initial conditions for a field."""
        ...

    def add_bc(self, bc: IBoundaryCondition):
        """Add a boundary condition."""
        ...

    def clear_bcs(self, field: str):
        """Clear boundary conditions for a field."""
        ...

    def add_callback(self, cb: "ISolverCallback"):
        """Add a callback."""
        ...

    def remove_callback(self, cb_id: str):
        """Remove a callback."""
        ...

    # -- lifecycle ----------------------------------

    def initialize(self, boundaries: IBoundaryProvider = None):
        """Cold setup for a run — binding phase, allowed to be expensive.

        Binds the boundary provider, allocates/refreshes the DataHub,
        applies ICs, sets time to t0. MAY re-load parameters from
        config; estimators must therefore NEVER call initialize
        between trials — they call reset(), which preserves the
        current parameter values.
        """
        ...

    def step(self, dt: float = None) -> SolverStatus:
        """Advance one step.

        Internal order: evaluate provider -> scatter-write
        value constraints -> publish flux products
        -> operators' forward -> time advance.
        """
        ...

    def reset(self):
        """Cheap rewind to the post-initialize state
        (estimation hot path, called once per trial).

        Preserves ALL bindings (boundary provider, DataHub allocation)
        and CURRENT parameter values — including trial parameters just
        set by an estimator. Re-applies ICs, clears history/caches,
        resets time counters. Under TRAIN it must leave a clean
        autograd graph (no stale cached tensors).

        Restoring an ARBITRARY state (hotstart / checkpoint) is NOT
        reset's job — that goes to the capability interfaces
        ISnapshotable.load (solver layer) / IStateful.restore (model
        layer)."""
        ...


# ---------------------------------------------------
# region Capability
# ---------------------------------------------------


class ISnapshotable(ABC):
    """Capability: solver snapshot.

    Snapshot = config + runtime state + model_refs. Parameter weights
    are NEVER embedded. load() must raise clear error listing missing
    model_id@version when resolution fails.
    """

    @abstractmethod
    def save(self, path: str): ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ISnapshotable": ...


class IAssimilatable(ABC):
    """Capability: merge external observations into runtime state.
    Assimilation adjusts STATE; estimation adjusts
    PARAMETERS — online correction of neural operators goes here."""

    @abstractmethod
    def assimilate(self, observations: Any, **kwargs): ...
