# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for callback classes used in the solvers.
"""
from core.numerics.mesh import Mesh
from abc import ABC, abstractmethod


class ISolverCallback(ABC):
    """
    Interface for callback class used in solving process.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Get the unique name of the callback method.
        """
        pass

    @abstractmethod
    def setup(self):
        """
        Set up the callback method.
        """
        pass

    @abstractmethod
    def on_task_begin(self):
        """
        Callback function called at the beginning of the task.
        """
        pass

    @abstractmethod
    def on_task_end(self):
        """
        Callback function called at the end of task.
        """
        pass

    @abstractmethod
    def on_step_begin(self):
        """
        Callback function called at step begin.
        """
        pass

    @abstractmethod
    def on_step(self):
        """
        Callback function called at each step.
        """
        pass

    @abstractmethod
    def on_step_end(self):
        """
        Callback function called at the end of step.
        """
        pass
