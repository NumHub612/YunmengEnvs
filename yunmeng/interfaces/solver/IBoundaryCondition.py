# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Boundary condition protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from typing import Protocol, runtime_checkable
from enum import Enum

from yunmeng.interfaces.supports import Region
from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region Boundary values
# ---------------------------------------------------


class BoundaryChannel(Enum):
    """How prescribed values enter the system."""

    CONSTRAINTS = "constraints"
    """Dirichlet class: scatter-WRITE values into the solution."""

    FLUXES = "fluxes"
    """Neumann class: scatter-ADD flux contributions."""

    MIXED = "mixed"
    """Robin class: coefficient pairs (reserved)."""


@dataclass
class BoundaryValues:
    """Per-instant boundary condition evaluation, backend arrays,
    graph-carrying allowed.
    """

    time: float

    #: var -> (field, [element_ids, prescribed values]); scatter-write (Dirichlet)
    constraints: dict[str, tuple[ArrayLike, ArrayLike]] = dc_field(default_factory=dict)

    #: var -> (field, [element_ids, flux contributions]); scatter-add (Neumann)
    fluxes: dict[str, tuple[ArrayLike, ArrayLike]] = dc_field(default_factory=dict)

    #: var -> (field, [element_ids, coeffs]); Robin/mixed, reserved
    mixed: dict[str, tuple[ArrayLike, ArrayLike]] = dc_field(default_factory=dict)

    def segment(self, channel: BoundaryChannel) -> dict:
        """The per-variable dict for one channel."""
        return {
            BoundaryChannel.CONSTRAINTS: self.constraints,
            BoundaryChannel.FLUXES: self.fluxes,
            BoundaryChannel.MIXED: self.mixed,
        }[channel]


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
    def target_field(self) -> str:
        """The name of field to which this BC applies."""
        ...

    @property
    @abstractmethod
    def region(self) -> Region:
        """The mesh region to which this BC applies."""
        ...

    @property
    @abstractmethod
    def channel(self) -> BoundaryChannel:
        """The channel to which this BC applies."""
        ...

    @abstractmethod
    def evaluate(self, t: float) -> ArrayLike:
        """Evaluate the rule at time t.

        Returns values ONLY, in the exact order of
        region.element_ids (invariant 4). Backend array,
        graph-carrying allowed.
        """
        ...


# ---------------------------------------------------
# region IBoundaryProvider
# ---------------------------------------------------


@runtime_checkable
class IBoundaryProvider(Protocol):
    """Per-instant boundary evaluator, held by model/solver.

    Merge-time validation: per-variable face_ids must be
    pairwise disjoint — overlapping segments are errors;
    uncovered faces mean natural boundary
    (zero normal gradient), which is a default,
    not an error.
    """

    @property
    @abstractmethod
    def rules(self) -> list[IBoundaryCondition]:
        """The list of boundary conditions."""
        ...

    @abstractmethod
    def evaluate(self, t: float) -> BoundaryValues:
        """Merge all boundary bindings at time t."""
        ...
