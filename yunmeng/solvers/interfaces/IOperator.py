# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for pde numerical operators.
"""

from yunmeng.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from yunmeng.numerics.mats.linalgs import LinearEqs
from yunmeng.numerics.fields.fields import Field
from yunmeng.numerics.fields.datahubs import DataHub
from yunmeng.numerics.mesh.spatials import Mesh
from abc import ABC, abstractmethod
import enum


class OperatorType(enum.Enum):
    """The operator type."""

    LAPLACIAN = "lap"
    DIV = "div"
    GRAD = "grad"
    D2DT2 = "d2dt2"
    DDT = "ddt"
    CURL = "curl"
    FUNC = "func"
    SRC = "src"
    LIMITER = "lim"
    UNKNOWN = "unknown"


class IOperator(ABC):
    """
    Interface for discretizing PDE term to computable form.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        The unique name of the operator.
        """
        pass

    @classmethod
    @abstractmethod
    def get_type(cls) -> OperatorType:
        """
        The type of the operator.
        """
        pass

    @property
    def time_order(self) -> int:
        """
        The time order of the operator.
        """
        return 1

    @abstractmethod
    def prepare(
        self,
        mesh: Mesh,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        """
        Prepare the operator.
        """
        pass

    @abstractmethod
    def run(self, sources: DataHub, timestep: float) -> Field | LinearEqs:
        """
        Run the operator on field.

        For explicit operators, it returns a `Field` for the updated field.
        For implicit operators, it returns a `LinearEqs` to be solved.
        """
        pass
