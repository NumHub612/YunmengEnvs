# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  

Custom defined boundary condition.
"""
from core.solvers.interfaces import IBoundaryCondition
from core.numerics.mesh import Node, Face, Cell


class CustomBoundary(IBoundaryCondition):
    """
    Custom defined boundary condition.
    """

    def __init__(self):
        pass

    @classmethod
    def get_name(cls):
        return "custom"

    def evalute(self, t: float, elem: Node | Face | Cell) -> tuple:
        """
        Evaluate the boundary condition at time t.

        Args:
            t (float): Time at which the boundary condition is evaluated, in seconds.

        Returns:
            The boundary condition at time t.
        """
        pass
