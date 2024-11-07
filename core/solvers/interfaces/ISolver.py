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
    def status(self) -> dict:
        """
        The current status of the solver, not including the solution.
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
        """
        pass

    @abstractmethod
    def set_ic(self, var: str, ic: IInitCondition):
        """
        Set the initial condition.
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
    def initialize(self, **kwargs):
        """
        Initialize the solver.
        """
        pass

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the solver.
        """
        pass

    @abstractmethod
    def terminate(self, **kwargs):
        """
        Terminate the solver.
        """
        pass

    @abstractmethod
    def optimize(self, **kwargs):
        """
        Optimize the solver with data to calibrate the parameters.
        """
        pass

    @abstractmethod
    def inference(self, **kwargs):
        """
        Inference the solver to get the solutions.
        """
        pass
