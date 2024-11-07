# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Interface for callback classes used in the solvers.
"""
from abc import ABC, abstractmethod


class ISolverCallback(ABC):
    """
    Interface for callback class for reporting solving process.
    """

    @abstractmethod
    def setup(self, solver_meta: dict):
        """
        Sets the solver metadata.
        """
        pass

    @abstractmethod
    def on_task_begin(self, **kwargs):
        """
        Callback function called at the beginning of task.
        """
        pass

    @abstractmethod
    def on_task_end(self, **kwargs):
        """
        Callback function called at the end of task.
        """
        pass

    @abstractmethod
    def on_step_begin(self, **kwargs):
        """
        Callback function called at the beginning of each step.
        """
        pass

    @abstractmethod
    def on_step(self, **kwargs):
        """
        Callback function called at each step.
        """
        pass

    @abstractmethod
    def on_step_end(self, **kwargs):
        """
        Callback function called at the end of each step.
        """
        pass
