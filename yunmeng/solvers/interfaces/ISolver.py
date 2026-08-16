# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for fluid equations solvers.
"""

from yunmeng.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from yunmeng.solvers.interfaces.IInitCondition import IInitialCondition
from yunmeng.solvers.interfaces.ISolverCallback import ISolverCallback
from yunmeng.solvers.interfaces.IEquation import IEquation
from yunmeng.numerics.enums import ElementType, MeshDimension, VariableType
from yunmeng.numerics.fields import Field, FieldMeta

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields as dc_fields
import enum
from typing import Any
import numpy as np

# --------------------------------------------------
# region Meta Description
# --------------------------------------------------


class SolverType(enum.Enum):
    """The solver type."""

    FDM = "fdm"  # Finite Difference Model.
    FVM = "fvm"  # Finite Volume Model.
    FEM = "fem"  # Finite Element Model.
    LBM = "lbm"  # Lattice Boltzmann Model.
    AIM = "aim"  # AI Model.
    UNKNOWN = "unknown"


@dataclass
class SolverMeta:
    """
    The meta description of the solver.
    """

    description: str = ""  # Brief description about the solver.
    type: SolverType = SolverType.UNKNOWN  # The solver type.
    equation: str = ""  # The equation to be solved, e.g. Swe2D.
    equation_expr: str = ""  # The mathematical expression.
    dimension: MeshDimension = MeshDimension.NONE
    default_ics: dict[str, IInitialCondition] = None
    default_bcs: dict[str, IBoundaryCondition] = None
    fields: dict[str, FieldMeta] = None  # The available fields.


@dataclass
class SolverStatus:
    """
    The current status of the solver.
    """

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


# --------------------------------------------------
# region Static Config
# --------------------------------------------------


@dataclass
class SolverConfig(ABC):
    """
    Base class for all solver configurations.
    """

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SolverConfig":
        """
        Build a config instance from a plain dictionary.
        Unknown keys are silently ignored.
        """
        keys = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in d.items() if k in keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        result = {}
        for f in dc_fields(self):
            val = getattr(self, f.name)
            if isinstance(val, np.generic):
                val = val.item()
            result[f.name] = val
        return result

    @classmethod
    @abstractmethod
    def get_solver_name(cls) -> str:
        """The solver class name this config belongs to."""
        raise NotImplementedError()


# --------------------------------------------------
# region Solver
# --------------------------------------------------


class ISolver(ABC):
    """
    Interface for fluid dynamic equations solvers.

    Four-layer lifecycle:

        1. Construction: __init__(id, mesh, operators, config)
        2. Assembly: add_*, remove_*, clear_* (repeatable)
        3. Initialize: initialize(...) (once)
        4. Runtime: forward(...) (repeatedly)
    """

    # -- class metadata -----------------------------

    @classmethod
    @abstractmethod
    def get_meta(cls) -> SolverMeta:
        """
        The meta infomations of solver.
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Unique name of this solver.
        """
        pass

    @classmethod
    @abstractmethod
    def get_config_class(cls) -> type[SolverConfig]:
        """
        The config class.
        """
        pass

    # -- properties ---------------------------------

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The solver id.
        """
        pass

    @property
    @abstractmethod
    def status(self) -> SolverStatus:
        """
        The current status.
        """
        pass

    @property
    @abstractmethod
    def config(self) -> SolverConfig:
        """
        The solver config.
        """
        pass

    # -- assembly -----------------------------------

    @abstractmethod
    def set_problems(self, equations: list[IEquation]):
        """
        Set the equations required to be solved.
        """
        pass

    @abstractmethod
    def get_solution(self, field: str) -> Field:
        """
        Get the solution of the solver.
        """
        pass

    @abstractmethod
    def add_ic(self, ic: IInitialCondition):
        """
        Add an initial condition.
        """
        pass

    @abstractmethod
    def clear_ics(self, field: str = None):
        """
        Clear initial conditions.
        """
        pass

    @abstractmethod
    def add_bc(self, bc: IBoundaryCondition):
        """
        Add a boundary condition.
        """
        pass

    @abstractmethod
    def clear_bcs(self, field: str):
        """
        Clear boundary conditions.
        """
        pass

    @abstractmethod
    def add_callback(self, cb: ISolverCallback):
        """
        Add a solver callback.
        """
        pass

    @abstractmethod
    def remove_callback(self, cb_id: str):
        """
        Remove a solver callback.
        """
        pass

    # -- lifecycle ----------------------------------

    @abstractmethod
    def initialize(self, **kwargs):
        """
        Initialize/reset the solver.
        """
        pass

    @abstractmethod
    def assimilate(self, **kwargs):
        """
        Assimilate with extra data.
        """
        pass

    @abstractmethod
    def forward(self, **kwargs) -> SolverStatus:
        """
        Advance solver to next timestep.
        """
        pass

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset solver to spcified state.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save solver snapshot to a file.
        """
        pass

    @classmethod
    @abstractmethod
    def load(self, path: str) -> "ISolver":
        """
        Load this solver from the file.
        """
        pass
