# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface of callback classes used in solvers.
"""
from core.numerics.mesh.spatials import Mesh
from abc import ABC, abstractmethod


class ISolverCallback(ABC):
    """
    Interface of callback class.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Get the unique name of this method.
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
    def setup(self, solver, mesh: Mesh, **kwargs):
        """
        Set up the callback method.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean the callback method.
        """
        pass

    @abstractmethod
    def on_task_begin(self):
        """
        Function called at the task begin.
        """
        pass

    @abstractmethod
    def on_task_end(self):
        """
        Function called at the task end.
        """
        pass

    @abstractmethod
    def on_step_begin(self):
        """
        Function called at step begin.
        """
        pass

    @abstractmethod
    def on_step(self):
        """
        Function called during step.
        """
        pass

    @abstractmethod
    def on_step_end(self):
        """
        Function called at the step end.
        """
        pass
