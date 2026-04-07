# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for fluid equations solvers.
"""
from yunmeng.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from yunmeng.solvers.interfaces.IInitCondition import IInitCondition
from yunmeng.solvers.interfaces.ISolverCallback import ISolverCallback
from yunmeng.solvers.interfaces.IEquation import IEquation
from yunmeng.numerics.enums import ElementType, MeshDimension, VariableType
from yunmeng.numerics.fields.fields import Field, FieldMeta
from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum
from typing import Any


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
    default_ics: dict[str, IInitCondition] = None
    default_bcs: dict[str, IBoundaryCondition] = None
    fields: dict[str, FieldMeta] = None  # The available fields.


@dataclass
class SolverStatus:
    """
    The current status of the solver.
    """

    finished: bool = False  # Whether the solver has finished.
    current_time: float = None  # Current physical time.
    end_time: float = None  # End time.
    time_step: float = None  # Current time step.
    residual: float = None  # Current step residual.
    step_time: float = None  # Current step time.
    total_time: float = None  # Total elapsed time.
    iteration: int = None  # Current iteration.
    error_code: int = 0  # 0 for no error,non-zero for errors.
    error_message: str = None  # Error message.
    etc: Any = None  # Any extra information.


class ISolver(ABC):
    """
    Interface for fluid dynamic equations solvers.
    """

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
    def add_callback(self, cb: ISolverCallback):
        """
        Add a callback to be called in solver.
        """
        pass

    @abstractmethod
    def add_ic(self, field: str, ic: IInitCondition):
        """
        Add an initial condition.
        """
        pass

    @abstractmethod
    def add_bc(
        self,
        field: str,
        bc: IBoundaryCondition,
        eids: list[int],
        etype: ElementType,
    ):
        """
        Add a boundary condition for the solver.
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Initialize/reset the solver.
        """
        pass

    @abstractmethod
    def assimilate(self, data: dict):
        """
        Assimilate with extra data.
        """
        pass

    @abstractmethod
    def inference(self) -> SolverStatus:
        """
        Advance the solver to the next timestep.
        """
        pass
