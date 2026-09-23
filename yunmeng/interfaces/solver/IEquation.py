# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Equation protocols.

Relationship between IEquation, IOperator and ISolver:

- Most solvers are purpose-built for a FIXED, known equation — their
  operator assembly is hard-coded (a Saint-Venant solver knows its
  terms). They never see IEquation.

- IEquation exists for CONFIGURABLE solvers: the equation is edited
  at run time through external configuration (teaching setups, term
  sensitivity experiments:add/remove a diffusion or source term and
  re-run). The solver calls set_problems([...]) before initialize().

Design invariants of the equation layer:

1. Equations DECLARE, operators DISCRETIZE. An IEquation is a pure
   declaration — which terms, acting on which fields, with which
   coefficient symbols. It never touches meshes, matrices or
   ILinearEqs; discretization lives exclusively in the operator layer.
   compile() is the ONLY bridge between the two.

2. Construction is complete. There are no phased setters: terms and
   symbols are fixed at construction time, validate() reports what's
   wrong, compile() refuses an invalid equation. A half-applied
   equation is unrepresentable by design.

3. Whole-assembly replacement. set_problems() replaces the solver's
   operator list ATOMICALLY: clear_operators() + add_operator() for
   each compiled operator. Per-term mixing of equation-produced
   operators with built-in ones is forbidden — a solver must never
   silently merge the two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field

from yunmeng.interfaces.capabilities import ParamMeta
from yunmeng.interfaces.supports import IBackend, IMesh
from yunmeng.interfaces.solver.IOperator import IOperator

# ---------------------------------------------------
# region Terms
# ---------------------------------------------------


@dataclass(frozen=True)
class EqTerm:
    """One term of a configurable equation: an operator reference plus
    its bindings. Immutable — edit by rebuilding the equation."""

    operator: str
    """Registered operator name (e.g. "FvmDiffusionOp", "SrcPointOp")."""

    target_field: str
    """The field this term acts on (must appear in IEquation.fields)."""

    coefficient: str = ""
    """Symbol name from the equation's symbol table; empty means a
    dimensionless unit coefficient. The symbol's current value is bound
    into the compiled operator's theta channel."""

    params: dict = dc_field(default_factory=dict)
    """Operator-specific construction arguments (stencil choices,
    limiter settings, NN sizes, ...)."""


# ---------------------------------------------------
# region IEquation
# ---------------------------------------------------


class IEquation(ABC):
    """Declarative description of a configurable PDE problem.

    Pure data + validation + compilation. No mesh binding, no field
    binding, no discretization state — those are compile()'s inputs and
    the operator layer's business respectively.
    """

    # -- identity -----------------------------------

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique equation name."""
        ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    # -- declaration --------------------------------

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """The fields this equation advances."""
        ...

    @property
    @abstractmethod
    def symbols(self) -> dict[str, ParamMeta]:
        """Coefficient symbols, declared as ParamMeta."""
        ...

    @property
    @abstractmethod
    def terms(self) -> list[EqTerm]:
        """The declared terms."""
        ...

    # -- checking -----------------------------------

    @abstractmethod
    def validate(self) -> list[str]:
        """Static checks, framework-style: every term's target_field is
        declared, every coefficient refers to a known symbol, every
        operator reference is resolvable in the registry, term list is
        non-empty. Empty list means valid."""
        ...

    # -- the bridge ---------------------------------

    @abstractmethod
    def compile(self, mesh: IMesh, backend: IBackend) -> list[IOperator]:
        """Instantiate one operator per term against this mesh/backend.

        The returned operators are CONSTRUCTED but NOT built — the
        solver builds them through its own assembly path."""
        ...

    @abstractmethod
    def summary(self) -> str:
        """Human-readable rendering of the equation, e.g.
        "du/dt = nu * laplacian(u) + src(u)" — for logs and reports."""
        ...
