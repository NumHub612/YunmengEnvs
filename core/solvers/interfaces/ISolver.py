# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Interfaces for fluid equations solvers.
"""
from core.numerics.fields import Field
from core.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from core.solvers.interfaces.IInitCondition import IInitCondition

from abc import ABC, abstractmethod


class ISolver(ABC):
    """
    Interface for fluid dynamic equations solvers.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The unique id of the solver instance.
        """
        pass

    @property
    @abstractmethod
    def current_time(self) -> float:
        """
        The current time of the solver.
        """
        pass

    @property
    @abstractmethod
    def total_time(self) -> float:
        """
        The total time of the solver.
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
    def get_meta(cls) -> dict:
        """
        The accessiable fields and other meta infomations of solver.
        """
        return {}

    @abstractmethod
    def get_solution(self, field_name: str) -> Field:
        """
        Get the solution of the solver.

        Args:
            field_name: The name of the field to get the solution.

        Returns:
            The solution of the solver.
        """
        pass

    @abstractmethod
    def set_bc(self, var: str, elems: list, bc: IBoundaryCondition):
        """
        Set the boundary condition for the solver.

        Args:
            var: Name of the variable to set the boundary condition.
            elems: The list of elements to set on.
            bc: The boundary condition.
        """
        pass

    @abstractmethod
    def set_ic(self, var: str, ic: IInitCondition):
        """
        Set the initial condition.

        Args:
            var: Name of the variable to set the initial condition.
            ic: The initial condition.
        """
        pass

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """
        Initialize the solver.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Advance this solver to the next step to solve the equations.
        """
        pass
