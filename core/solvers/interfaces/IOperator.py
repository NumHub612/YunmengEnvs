# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for pde numerical operators.
"""
from core.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from core.numerics.mats.linalgs import LinearEqs
from core.numerics.fields.fields import Field
from core.numerics.fields.datahubs import DataHub
from core.numerics.mesh.spatials import Mesh
from abc import ABC, abstractmethod
import enum


class OperatorType(enum.Enum):
    """The operator type."""

    D2DT2 = "d2dt2"
    DIV = "div"
    GRAD = "grad"
    LAPLACIAN = "laplacian"
    DDT = "ddt"
    CURL = "curl"
    FUNC = "func"
    SRC = "src"
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
        fields: list[str],
        mesh: Mesh,
        bounds: dict[int, dict[str, IBoundaryCondition]],
    ):
        """
        Prepare the operator.
        """
        pass

    @abstractmethod
    def run(self, sources: DataHub) -> Field | LinearEqs:
        """
        Run the operator on mesh.
        """
        pass
