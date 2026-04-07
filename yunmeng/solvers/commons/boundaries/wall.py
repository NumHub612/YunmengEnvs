# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide the wall boundary condition.
"""
from yunmeng.solvers.interfaces import IBoundaryCondition, BoundaryType, BoundaryValue
from yunmeng.setting import logger


class WallBoundary(IBoundaryCondition):
    """
    Wall boundary condition for riverbed, embankment.
    Required:
    + extra={'slip': bool, 'roughness': float}
    """

    @classmethod
    def get_type(cls) -> BoundaryType:
        return BoundaryType.WALL

    @classmethod
    def get_name(cls) -> str:
        return "wall"

    def __init__(self, id: str, slip: bool = False, roughness: float = 0.0):
        self._id = id
        self._bc = BoundaryValue(
            extra={"slip": slip, "roughness": roughness},
        )

    @property
    def id(self) -> str:
        return self._id

    def update(self, slip: bool = None, roughness: float = None):
        """
        Update the boundary condition.

        Args:
            value: New boundary value.
        """
        if slip is not None:
            self._bc.extra["slip"] = slip
        if roughness is not None:
            self._bc.extra["roughness"] = roughness

        logger.info(f"Boundary {self.id} updated to {self._bc}.")

    def evaluate(self) -> BoundaryValue:
        return self._bc
