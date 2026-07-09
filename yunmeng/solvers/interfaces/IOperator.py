# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for pde numerical operators.
"""

from yunmeng.solvers.interfaces.IBoundaryCondition import IBoundaryCondition
from yunmeng.numerics.linalgs import LinearEqs
from yunmeng.numerics.fields import Field, DataHub2, DataProduct, Sample2
from yunmeng.numerics.mesh import Mesh, ElementType
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


class OperatorMode(enum.Enum):
    """The operator mode."""

    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    UNKNOWN = "unknown"


class IOperator(ABC):
    """
    Interface for discretizing PDE term to computable form.

    Lifecycle:
        1. Construction: ``__init__(fields, **kwargs)``
           and scheme-specific parameters (e.g. configures, limiter type).
        2. Preparation: ``prepare(mesh, bounds)`` — precompute stencils,
           neighbor indices, allocate scratch arrays. Called once before
           the time-stepping loop.
        3. Evaluation: ``forward(data_hub, **kwargs)``— called every step.
    """

    # -- class metadata -----------------------------

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

    @classmethod
    @abstractmethod
    def get_mode(cls) -> OperatorMode:
        """
        The mode of the operator.
        """
        pass

    # -- class -----------------------------

    @classmethod
    def produces(cls, field_name: str, etype: ElementType) -> list[DataProduct]:
        """
        Declares DataProducts this operator can produce for.
        """
        return []

    @classmethod
    def consumes(cls, field_name: str, etype: ElementType) -> list[DataProduct]:
        """
        Declares DataProducts this operator want to consume.
        """
        return []

    # -- fields -------------------------------------

    @property
    def target_fields(self) -> list[str]:
        """
        The operator target fields.
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
        boundaries: dict[str, list[IBoundaryCondition]],
        **kwargs,
    ):
        """
        Prepares the operator.
        """
        pass

    @abstractmethod
    def forward(self, datahub: DataHub2, **kwargs) -> Field | LinearEqs:
        """
        Runs the operator on field.

        For explicit operators, it returns a `Field`  generated newly.
        For implicit operators, it returns a `LinearEqs` to be solved.
        """
        pass
