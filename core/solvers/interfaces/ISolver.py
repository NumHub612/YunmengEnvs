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


class ISolver(ABC):
    """
    Interface for fluid dynamic equations solvers.
    """

    @classmethod
    @abstractmethod
    def get_meta(cls) -> dict:
        """
        The accessiable fields and other meta infomations of solver.

        Returns:
            - A dictionary containing the meta information of the solver with the following keys:
                - description (str): A brief description of the solver.
                - type (str): The type of the solver.
                - equation (str): The equation solved by the solver, e.g. Navier-Stokes, etc.
                - equation_expr (str): The mathematical expression of the equation.
                - dimension (str): The dimension of the solver.
                - default_ics (dict): The default initialization conditions.
                - default_bcs (dict): The default boundary conditions.
                - fields (dict): The dictionary of available fields solved by the solver.

        Notes:
            - The `fields` contains all the avaiable fields from the solver, following the format:
                - description (str): A brief description.
                - dtype (str): The data type, e.g. scalar, vector, tensor.
                - etype (str): The element type of the field, e.g. node, face, cell.
        """
        return {}

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Unique name of this solver.
        """
        pass

    @property
    @abstractmethod
    def status(self) -> dict:
        """
        The current status of the solver, not including the solution.

        Returns:
            - A dictionary containing the current status of the solver with the following keys:
                - elapsed_time (float): The elapsed time since the start of the solver.
                - iteration (int): The current iteration number.
                - time_step (float): The current time step.
                - current_time (float): The current time.
                - convergence (bool): Whether the solver has converged.
                - error (str): Any error messages or warnings.
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
    def inference(self) -> tuple[bool, bool, dict]:
        """
        Inference the solver to get the solutions.
        Run by steps.

        Returns:
            A tuple of results with (is_done, is_terminated, status).
        """
        pass
