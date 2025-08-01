# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interfaces for fluid equations solvers.
"""
from core.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from core.solvers.interfaces.IInitCondition import IInitCondition
from core.solvers.interfaces.ISolverCallback import ISolverCallback
from core.solvers.interfaces.IEquation import IEquation
from core.numerics.fields import Field
from abc import ABC, abstractmethod
from dataclasses import dataclass
import enum


class SolverType(enum.Enum):
    """The solver type."""

    FVM = "fvm"
    FDM = "fdm"
    FEM = "fem"
    LBM = "lbm"
    NN = "nn"
    HYB = "hyb"
    UNKNOWN = "unknown"


@dataclass
class SolverMeta:
    """
    The meta information of the solver.
    """

    description: str = ""  # A brief description about this solver.
    type: SolverType = SolverType.UNKNOWN  # The solver type.
    equation: str = ""  # The equation solved by the solver, e.g. Burgers, etc.
    equation_expr: str = ""  # The mathematical expression of the equation.
    dimension: str = ""  # The equation dimension, e.g. 1d, 2d, 3d.
    default_ics: dict = None  # Default initialization conditions.
    default_bcs: dict = None  # Default boundary conditions.
    fields: dict = None  # The dictionary of available fields solved.


@dataclass
class SolverStatus:
    """
    The current status of the solver, excluding the solutions.
    """

    elapsed_time: float = 0.0  # Time spent on the current step.
    residual: float = 0.0  # The max residual in current step.
    iteration: int = 0  # The iteration number.
    time_step: float = 0.0  # The current calculation time step.
    progress: float = 0.0  # Progress percentage (0~1).
    finished: bool = False  # Whether the solver has finished.
    converged: bool = False  # Whether the solver has converged.
    msg: str = ""  # Any info messages, errors or warnings.


class ISolver(ABC):
    """
    Interface for fluid dynamic equations solvers.
    """

    @classmethod
    @abstractmethod
    def get_meta(cls) -> SolverMeta:
        """
        The accessiable fields and other meta infomations of solver.

        Notes:
            - The `fields` contains all the avaiable fields with followings:
                - description (str): A brief description.
                - dtype (str): Data type, [scalar, vector, tensor].
                - etype (str): Element type, [node, face, cell].
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
        The current status of the solver, not including the solution.
        """
        pass

    @abstractmethod
    def get_solution(self, field_name: str) -> Field:
        """
        Get the solution of the solver.
        """
        pass

    @abstractmethod
    def set_problems(self, equations: list[IEquation]):
        """
        Set the equations to be solved by the solver.
        """
        pass

    @abstractmethod
    def add_callback(self, callback: ISolverCallback):
        """
        Add a callback to be called during the solver.
        """
        pass

    @abstractmethod
    def add_ic(self, var: str, ic: IInitCondition):
        """
        Add the initial condition.
        """
        pass

    @abstractmethod
    def add_bc(self, var: str, elements: list, bc: IBoundaryCondition):
        """
        Add the boundary condition for the solver.

        Args:
            var: Name of the variable to set the boundary condition.
            elements: The list of elements to set on.
            bc: The boundary condition.
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Initialize the solver.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the solver.
        """
        pass

    @abstractmethod
    def terminate(self):
        """
        Terminate the solver.
        """
        pass

    @abstractmethod
    def assimilate(self, data: dict):
        """
        Assimilate the solver with extra data to improve its accuracy.
        Run by steps.

        Args:
            data: The extra data to assimilate.
        """
        pass

    @abstractmethod
    def optimize(self):
        """
        Optimize the solver arguments to improve its accuracy, etc.

        Run by steps or in a batch.
        """
        pass

    @abstractmethod
    def inference(self) -> tuple[bool, bool, SolverStatus]:
        """
        Inference the solver to get the solutions.
        Run by steps.

        Returns:
            A tuple of results with (is_done, is_terminated, status).
        """
        pass
