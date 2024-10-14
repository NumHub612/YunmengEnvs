# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Interfaces for boundary conditions.
"""
from abc import ABC, abstractmethod


class IBoundaryCondition(ABC):
    """
    Interface for boundary conditions.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the boundary condition.
        """
        raise NotImplementedError

    @abstractmethod
    def evalute(self, **kwargs) -> tuple:
        """
        Evaluate the boundary condition.

        Returns:
            A tuple of (flux, value) conditions.
        """
        raise NotImplementedError
