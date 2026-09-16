# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solver-layer protocols.
"""

from __future__ import annotations

from dataclasses import dataclass

from yunmeng.interfaces.types import ArrayLike, ElementType, RunMode
from yunmeng.interfaces.support import (
    IBackend,
    IDataHub,
    DataProduct,
    IField,
    ILinearEqs,
    IMesh,
)

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
# region IModeSwitchable
# ---------------------------------------------------


@dataclass
class OperatorResult:
    """The result of an operator evaluation."""

    explicit: IField = None
    implicit: ILinearEqs = None


class IModeSwitchable:
    """Capability: mode-sensitive behavior.

    TRAIN contract: forward() must allocate fresh outputs;
    EVAL mode may reuse buffers.
    """

    def set_mode(self, mode: RunMode): ...


# ---------------------------------------------------
# region IOperator
# ---------------------------------------------------


class IOperator:
    """Operator discretizing PDE term to computable form.
    Runtime-pure and reusable acrossruns."""

    # -- class metadata -----------------------------

    @classmethod
    def get_name(cls) -> str:
        """The unique name of the operator."""
        ...

    @classmethod
    def get_kind(cls) -> str:
        """Open kind string."""
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

    # -- fields -------------------------------------

    @property
    def target_fields(self) -> list[str]:
        """The name of fields the operator acts on."""
        ...

    @property
    def time_order(self) -> int:
        """The time order of the operator."""
        return 1

    # -- lifecycle ----------------------------------

    def build(self, mesh: IMesh, backend: IBackend):
        """Static topology phase, ONCE per mesh: precompute stencils,
        neighbor indices, matrix structure, network shapes;
        move index arrays to the backend device."""
        ...

    def forward(
        self,
        datahub: IDataHub,
        t: float,
        dt: float,
    ) -> OperatorResult:
        """Runtime phase, EVERY step."""
        ...
