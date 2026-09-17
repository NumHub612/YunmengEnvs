# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solver-layer protocols.

solver layer is where DIFFERENTIABILITY lives: RunMode.TRAIN,
IDifferentiable (live autograd leaves) and the "no detach, no
stale-cache replay" contract apply to this layer and below (operators,
DataHub) — never to the model/exchange layer above.

Design invariants of the solver layer:

1. Inference-pure core: this interface carries no training concerns.
   Training capabilities (IEstimable, IDifferentiable, IModeSwitchable)
   are optional and isinstance-governed.

2. Gradient boundary: the autograd graph may span fields, operators and
   the DataHub inside one solver run, but terminates at the model
   boundary. Solvers therefore never see exchange items; they see
   fields, regions and DataProducts only.

3. TRAIN/EVAL switching is driven by the owning model
   (propagate_mode: model -> solver -> DataHub -> operators). A solver
   must assume standalone TRAIN context has already been validated by
   its owner; in coupled runs it will only ever observe EVAL.

4. reset() vs snapshot(): reset() is the cheap rewind to the
   post-initialize state for estimation trials (preserves bindings and
   trial parameters, leaves a clean autograd graph under TRAIN);
   arbitrary-state restore belongs to ISnapshottable. Under EVAL-only
   coupling, snapshots are pure state copies with no graph semantics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields as dc_fields
from typing import Any

from yunmeng.interfaces.supports import FieldMeta, IField
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


class ISolver(ABC):
    """Inference-time solver CORE interface.

    Lifecycle:
        1. Construction: class-specific, constrained at the
           factory/registry layer (get_name/get_config_class/get_meta),
           NEVER by this protocol.
        2. Assembly: add_*, remove_*, clear_* (repeatable)
        3. Initialize: initialize(...) (once per run)
        4. Runtime: step(dt) (repeatedly)
    """

    # -- class metadata -----------------------------

    @classmethod
    @abstractmethod
    def get_meta(cls) -> SolverMeta: ...

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique solver class name."""
        ...

    @classmethod
    @abstractmethod
    def get_config_class(cls) -> type[SolverConfig]: ...

    # -- properties ---------------------------------

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def status(self) -> SolverStatus: ...

    @property
    @abstractmethod
    def config(self) -> SolverConfig: ...

    @property
    @abstractmethod
    def mode(self) -> RunMode: ...

    # -- mode switching -----------------------------

    @abstractmethod
    def train(self):
        """Set TRAIN and propagate: solver -> DataHub -> operators."""
        ...

    @abstractmethod
    def eval(self):
        """Set EVAL and propagate (by default)."""
        ...

    # -- assembly -----------------------------------

    @abstractmethod
    def set_problems(self, eqs: list["IEquation"]):
        """Set the PDE problems to be solved."""
        ...

    @abstractmethod
    def get_solution(self, field: str) -> IField:
        """Get the solution field by name."""
        ...

    @abstractmethod
    def add_ic(self, ic: IInitialCondition):
        """Add an initial condition."""
        ...

    @abstractmethod
    def clear_ics(self, field: str = None):
        """Clear initial conditions for a field."""
        ...

    @abstractmethod
    def add_bc(self, bc: IBoundaryCondition):
        """Add a boundary condition."""
        ...

    @abstractmethod
    def clear_bcs(self, field: str):
        """Clear boundary conditions for a field."""
        ...

    @abstractmethod
    def add_callback(self, cb: "ISolverCallback"):
        """Add a callback."""
        ...

    @abstractmethod
    def remove_callback(self, cb_id: str):
        """Remove a callback."""
        ...

    # -- lifecycle ----------------------------------

    @abstractmethod
    def initialize(self, boundaries: IBoundaryProvider = None):
        """Cold setup for a run — binding phase, allowed to be expensive.

        Binds the boundary provider, allocates/refreshes the DataHub,
        applies ICs, sets time to t0. MAY re-load parameters from config.
        """
        ...

    @abstractmethod
    def step(self, dt: float = None) -> SolverStatus:
        """Advance one step.

        Internal order: evaluate provider -> scatter-write
        value constraints -> publish flux products
        -> operators' forward -> time advance.
        """
        ...

    @abstractmethod
    def reset(self):
        """Cheap rewind to the post-initialize state
        (estimation hot path, called once per trial).

        Preserves ALL bindings (boundary provider, DataHub allocation)
        and CURRENT parameter values — including trial parameters just
        set by an estimator. Re-applies ICs, clears history/caches,
        resets time counters. Under TRAIN it must leave a clean
        autograd graph (no stale cached tensors).

        Restoring an ARBITRARY state (hotstart / checkpoint) is NOT
        reset's job — that is capabilities.ISnapshottable's."""
        ...
