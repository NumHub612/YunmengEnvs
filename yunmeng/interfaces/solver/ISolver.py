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

4. Operator assembly is protocol-level. The operator list is mutable
   through add_operator/remove_operator/clear_operators — never by
   reaching into solver internals. build() timing rule: operators added
   BEFORE initialize() are built during initialize(); operators added
   AFTER initialize() are built immediately (hot add).

5. Equation binding: a solver declares in its
   SolverMeta whether its governing equation (and possibly its operator
   assembly) is FIXED at development time or CONFIGURABLE at run time.
   FIXED is the common case; set_problems() on a FIXED solver is
   rejected with a clear error. IEquation objects must never be
   silently half-applied.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields as dc_fields, asdict
from typing import Any

from yunmeng.interfaces.supports import FieldMeta, IField
from yunmeng.interfaces.solver.IBoundaryCondition import (
    IBoundaryProvider,
    IBoundaryCondition,
)
from yunmeng.interfaces.solver.IInitCondition import IInitialCondition
from yunmeng.interfaces.solver.IOperator import IOperator
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
        return asdict(self)

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
    def bcs(self) -> list[IBoundaryCondition]: ...

    @property
    @abstractmethod
    def ops(self) -> list[IOperator]: ...

    @property
    @abstractmethod
    def ics(self) -> list[IInitialCondition]: ...

    @property
    @abstractmethod
    def status(self) -> SolverStatus: ...

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

    # -- problem binding ----------------------------

    @abstractmethod
    def set_problems(self, eqs: list["IEquation"]):
        """Set the PDE problems to be solved."""
        ...

    @abstractmethod
    def get_solution(self, field: str) -> IField:
        """Get the solution field by name."""
        ...

    # -- assembly -----------------------------------

    @abstractmethod
    def add_callback(self, cb: "ISolverCallback"):
        """Add a callback."""
        ...

    @abstractmethod
    def remove_callback(self, cb_id: str):
        """Remove a callback."""
        ...

    @abstractmethod
    def add_op(self, op: "IOperator"):
        """Add an operator."""
        ...

    @abstractmethod
    def remove_op(self, op_id: str):
        """Remove an operator."""
        ...

    @abstractmethod
    def add_ic(self, ic: IInitialCondition):
        """Add an initial condition."""
        ...

    @abstractmethod
    def remove_ic(self, ic_id: str):
        """Remove initial condition."""
        ...

    @abstractmethod
    def add_bc(self, bc: IBoundaryCondition):
        """Add a boundary condition."""
        ...

    @abstractmethod
    def remove_bc(self, bc_id: str):
        """Remove boundary condition."""
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
