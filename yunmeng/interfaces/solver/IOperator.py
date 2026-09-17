# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solver-layer protocols.

Operators discretize PDE terms — mathematical, source, hydraulic-
structure AND neural — into computable form. Physical and neural
operators are first-class citizens of the SAME interface
(OperatorKinds: math.* / src.* / structure.* / nn.*); the mixing ratio
of physics to AI is a user assembly choice, not a framework branch.

Design invariants of the operator layer:

1. Differentiability is OPTIONAL, never an entry requirement.
   Stateless math operators (e.g. FDM stencils, grad/div/laplacian)
   carry no parameters and no graph obligations. Graph duties arise ONLY
   when an operator opts in: declaring differentiable=True or
   implementing IDifferentiable commits its forward() to the TRAIN
   contract (fresh outputs, no detach, no stale-cache replay) and to
   exposing live autograd leaves.

2. Non-participation is not sabotage. Under TRAIN, an operator that
   does not join the graph must still be graph-NEUTRAL: it must not
   detach, copy-detach or otherwise sever gradients on values passing
   through it — inputs from the DataHub may carry a graph owned by
   others. If an operator cannot guarantee neutrality, it must say so
   via supports_gradients()=False and let the caller decide.

3. Two computation phases: build(mesh, backend) runs ONCE per mesh
   (stencils, neighbor indices, matrix structure, network shapes,
   device placement); forward(datahub, t, dt) runs EVERY step and must
   stay allocation-lean and backend-agnostic — array math goes through
   IBackend primitives, never through direct library calls.

4. Training capabilities are optional and isinstance-governed:
   IParameterized (copy-semantic theta channel for archival and
   gradient-free calibration), IDifferentiable (live leaves),
   IModeSwitchable (TRAIN/EVAL behavior). A stateless operator
   typically implements none of them.

5. Explicit/implicit duality: an operator declares explicit_part and/or
   implicit_part and returns OperatorResult accordingly; implicit
   contributions are handed to the solver as ILinearEqs, and sparse/
   iterative solves on the gradient path must be wrapped via the
   implicit-function theorem (see IBackend.solve), never raw AD
   through iterations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from yunmeng.interfaces.types import ElementType
from yunmeng.interfaces.supports import (
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


@dataclass
class OperatorResult:
    """The result of an operator evaluation."""

    explicit: IField = None
    implicit: ILinearEqs = None


# ---------------------------------------------------
# region IOperator
# ---------------------------------------------------


class IOperator(ABC):
    """Operator discretizing PDE term to computable form.
    Runtime-pure and reusable across runs.

    Optional capabilities (structural, runtime_checkable Protocols):
      IParameterized   — θ channel (copy semantics)
      IDifferentiable  — live autograd leaves for training
      IModeSwitchable  — TRAIN/EVAL-sensitive behavior
    """

    # -- class metadata -----------------------------

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique name of the operator."""
        ...

    @classmethod
    @abstractmethod
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
    @abstractmethod
    def explicit_part(self) -> bool:
        """Produces an explicit Field contribution."""
        ...

    @property
    @abstractmethod
    def implicit_part(self) -> bool:
        """Produces a `LinearEqs` to be assembled."""
        ...

    @property
    @abstractmethod
    def differentiable(self) -> bool:
        """Is the operator differentiable."""
        ...

    # -- fields -------------------------------------

    @property
    @abstractmethod
    def target_fields(self) -> list[str]:
        """The name of fields the operator acts on."""
        ...

    @property
    def time_order(self) -> int:
        """The time order of the operator."""
        return 1

    # -- lifecycle ----------------------------------

    @abstractmethod
    def build(self, mesh: IMesh, backend: IBackend):
        """Static topology phase, ONCE per mesh: precompute stencils,
        neighbor indices, matrix structure, network shapes;
        move index arrays to the backend device."""
        ...

    @abstractmethod
    def forward(
        self,
        datahub: IDataHub,
        t: float,
        dt: float,
    ) -> OperatorResult:
        """Runtime phase, EVERY step."""
        ...
