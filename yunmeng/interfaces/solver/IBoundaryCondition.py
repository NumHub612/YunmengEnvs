# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Boundary condition protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field

from yunmeng.interfaces.supports import Region
from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region Boundary values
# ---------------------------------------------------


@dataclass
class BoundaryValues:
    """Per-instant boundary condition evaluation, backend arrays,
    graph-carrying allowed.
    """

    time: float

    #: var -> (face_ids, prescribed values); scatter-write (Dirichlet)
    constraints: dict[str, tuple[ArrayLike, ArrayLike]] = dc_field(default_factory=dict)

    #: var -> (face_ids, flux contributions); scatter-add (Neumann)
    fluxes: dict[str, tuple[ArrayLike, ArrayLike]] = dc_field(default_factory=dict)

    #: var -> (face_ids, coeffs); Robin/mixed, reserved
    mixed: dict[str, tuple[ArrayLike, ArrayLike]] = dc_field(default_factory=dict)


# ---------------------------------------------------
# region IBoundaryCondition
# ---------------------------------------------------


class IBoundaryCondition(ABC):
    """Boundary rule object."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique name of the boundary condition."""
        ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def semantic_tag(self) -> str:
        """Free-form type tag, e.g. "wall", "open"."""
        ...

    @property
    @abstractmethod
    def target_field(self) -> str:
        """The name of field to which this BC applies."""
        ...

    @property
    @abstractmethod
    def region(self) -> Region:
        """The mesh region to which this BC applies."""
        ...

    @abstractmethod
    def evaluate(self, t: float) -> tuple[str, ArrayLike]:
        """Evaluate the rule at time t.

        Returns (channel, values) where
        channel is one of "constraints"|"fluxes"|"mixed"
        and values are backend
        arrays aligned with region.element_ids.
        """
        ...


# ---------------------------------------------------
# region IBoundaryProvider
# ---------------------------------------------------


class IBoundaryProvider(ABC):
    """Per-instant boundary evaluator, held by model/solver."""

    @abstractmethod
    def evaluate(self, t: float) -> BoundaryValues:
        """Merge all (region, variable, rule) bindings at time t.

        Merge-time validation: per-variable face_ids must
        be pairwise disjoint — overlapping segments are errors;
        uncovered faces mean natural boundary
        (zero normal gradient), which is a default, not an error.
        """
        ...
