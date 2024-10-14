# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Interfaces for fluid equations solvers.
"""
from abc import ABC, abstractmethod


class ISolver(ABC):
    """
    Interface for fluid dynamic equations solvers.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the solver.
        """
        pass

    @abstractmethod
    def update(self, **kwargs):
        """
        Update the solver to get the new solution.
        """
        pass
