# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for pde numerical operators.
"""
from core.numerics.mats import LinearEqs
from core.numerics.fields import Field
from core.numerics.mesh import Mesh
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
    Interface for discretizing pde term to a numerical form.
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
    def get_type(self) -> OperatorType:
        """
        The type of the operator.
        """
        pass

    @abstractmethod
    def prepare(self, mesh: Mesh):
        """
        Prepare the operator.
        """
        pass

    @abstractmethod
    def run(self, source: Field) -> Field | LinearEqs:
        """
        Run the operator on mesh.
        """
        pass
