# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, share your ideas!  

Interface for callback classes used in the solvers.
"""
from core.numerics.mesh import Mesh
from abc import ABC, abstractmethod


class ISolverCallback(ABC):
    """
    Interface for callback class for reporting solving process.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Get the unique name of the callback method.
        """
        pass

    @abstractmethod
    def setup(self, solver_meta: dict, mesh: Mesh):
        """
        Set the solver meta information and mesh.
        """
        pass

    @abstractmethod
    def on_task_begin(self, solver_status: dict, solver_solutions: dict):
        """
        Callback function called at the beginning of the task.
        """
        pass

    @abstractmethod
    def on_task_end(self, solver_status: dict, solver_solutions: dict):
        """
        Callback function called at the end of task.
        """
        pass

    @abstractmethod
    def on_step_begin(self, solver_status: dict, solver_solutions: dict):
        """
        Callback function called at step begin.
        """
        pass

    @abstractmethod
    def on_step(self, solver_status: dict, solver_solutions: dict):
        """
        Callback function called at each step.
        """
        pass

    @abstractmethod
    def on_step_end(self, solver_status: dict, solver_solutions: dict):
        """
        Callback function called at the end of step.
        """
        pass
