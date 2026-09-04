# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solver-layer protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from yunmeng.interfaces.support.backend import Backend
from yunmeng.interfaces.support.datahub import IDataHub
from yunmeng.interfaces.support.field import DataProduct, IField
from yunmeng.interfaces.support.linalg import ILinearEqs
from yunmeng.interfaces.support.mesh import IMesh
from yunmeng.interfaces.types import ArrayLike, ElementType, RunMode

# ---------------------------------------------------
# region Operator kinds
# ---------------------------------------------------


class OperatorKinds:
    """Well-known operator kind constants ("<domain>.<name>")."""

    # mathematical discretizations
    MATH_GRAD = "math.grad"
    MATH_DIV = "math.div"
    MATH_LAPLACIAN = "math.laplacian"
    MATH_CURL = "math.curl"
    MATH_DDT = "math.ddt"
    MATH_D2DT2 = "math.d2dt2"
    MATH_LIMITER = "math.limiter"

    # sources
    SRC_POINT = "src.point"
    SRC_FUNC = "src.func"

    # hydraulic structures
    STRUCTURE_WEIR = "structure.weir"
    STRUCTURE_GATE = "structure.gate"
    STRUCTURE_ORIFICE = "structure.orifice"
    STRUCTURE_PUMP = "structure.pump"

    # neural operators
    NN_CORRECTION = "nn.correction"
    NN_CLOSURE = "nn.closure"
    NN_SURROGATE = "nn.surrogate"

    _CORE = frozenset(
        v for k, v in vars().items() if k.isupper() and isinstance(v, str)
    )


def is_known_kind(kind: str) -> bool:
    """Diagnostic helper: is *kind* a framework-known core kind?"""
    return kind in OperatorKinds._CORE


# ---------------------------------------------------
# region result & capability
# ---------------------------------------------------


@dataclass
class OperatorResult:
    """The result of an operator evaluation."""

    explicit: IField = None
    implicit: ILinearEqs = None


@runtime_checkable
class IParameterized(Protocol):
    """Capability: parameters θ io channel."""

    def get_parameters(self) -> dict[str, ArrayLike]: ...

    def set_parameters(
        self,
        params: Mapping[str, ArrayLike],
    ): ...


@runtime_checkable
class IModeSwitchable(Protocol):
    """Capability: mode-sensitive behavior.

    TRAIN contract: forward() must allocate fresh outputs;
    EVAL mode may reuse buffers.
    """

    def set_mode(self, mode: RunMode) -> None: ...


# ---------------------------------------------------
# region IOperator
# ---------------------------------------------------


class IOperator(Protocol):
    """Operator discretizing PDE term to computable form.
    Runtime-pure and reusable acrossruns."""

    # -- class metadata -----------------------------

    @classmethod
    def get_name(cls) -> str:
        """The unique name of the operator."""
        ...

    @classmethod
    def get_kind(cls) -> str:
        """Open kind string, see `OperatorKinds`."""
        ...

    # -- structural behavior flags ------------------

    @property
    def explicit_part(self) -> bool:
        """Produces an explicit Field contribution."""
        ...

    @property
    def implicit_part(self) -> bool:
        """Produces a `LinearEqs` to be assembled."""
        ...

    @property
    def differentiable(self) -> bool:
        """Is the operator differentiable."""
        ...

    # -- products -----------------------------------

    @classmethod
    def produces(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        """Declares DataProducts this operator can produce for."""
        return []

    @classmethod
    def consumes(cls, fname: str, loc: ElementType) -> list[DataProduct]:
        """Declares DataProducts this operator can consume."""
        return []

    # -- fields -------------------------------------

    @property
    def target_fields(self) -> Sequence[str]:
        """The operator target fields."""
        ...

    @property
    def time_order(self) -> int:
        """The time order of the operator."""
        return 1

    # -- lifecycle ----------------------------------

    def build(self, mesh: IMesh, backend: Backend):
        """Static topology phase, ONCE per mesh: precompute stencils,
        neighbor indices, matrix structure, network shapes;
        move index arrays to the backend device.
        MUST NOT load parameters θ and MUST NOT bind runtime data."""
        ...

    def forward(self, datahub: IDataHub, t: float, dt: float) -> OperatorResult:
        """Runtime phase, EVERY step. Boundary data is read from
        fields (value constraints already scatter-written by the
        solver) and from hub products under KEY_BOUNDARY (flux
        constraints)."""
        ...
