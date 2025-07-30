# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for callback classes used in the solvers.
"""
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

    @property
    @abstractmethod
    def id(self) -> str:
        """
        The callback id.
        """
        pass

    @abstractmethod
    def setup(self, solver: object, mesh: object, **kwargs):
        """
        Set up the callback method.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up the callback method.
        """
        pass

    @abstractmethod
    def on_task_begin(self):
        """
        Callback function called at the task begin.
        """
        pass

    @abstractmethod
    def on_task_end(self):
        """
        Callback function called at the task end.
        """
        pass

    @abstractmethod
    def on_step_begin(self):
        """
        Callback function called at the step begin.
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
        Callback function called at the step end.
        """
        pass
