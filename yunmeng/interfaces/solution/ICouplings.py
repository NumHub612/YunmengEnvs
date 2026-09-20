# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Coupling strategy protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any

from yunmeng.interfaces.solution.IModel import ILinkableModel
from yunmeng.interfaces.types import ArrayLike

# ---------------------------------------------------
# region Coupling kinds
# ---------------------------------------------------


class CouplingKinds:
    """Well-known coupling kind constants."""

    PULL = "pull"
    """One-way data pull (OpenMI-style). Downstream Model reads
    from upstream output when it updates."""

    LOOP = "loop"
    """Iterative two-way coupling.Both components exchange data 
    and iterate within a single time step until convergence."""

    PUSH = "push"
    """One-way data push (rarely used in practice)."""

    AGENT = "agent"
    """Agent-driven control coupling. An `IAgentModel` observes 
    components and injects actions back into the system."""

    SURROGATE = "surrogate"
    """Surrogate coupling. A AI model replaces or accelerates 
    an expensive physics model while keeping the same ports."""

    NESTED = "nested"
    """Nested/multi-scale coupling: children run at finer time
    resolution inside one parent step and feed aggregates back."""

    _CORE = frozenset(
        v for k, v in vars().items() if k.isupper() and isinstance(v, str)
    )


class DivergenceAction(Enum):
    """Action when LOOP iteration finishes without convergence."""

    ROLLBACK = "rollback"
    """restore pre-iteration state and mark FAILED"""

    CONTINUE = "continue"
    """accept the best approximation and emit a warning"""

    FREEZE = "freeze"
    """keep the last converged state from previous step"""


@dataclass
class CouplingConfig:
    """Configuration for one coupling link."""

    mode: str = CouplingKinds.PULL

    # -- LOOP-only parameters -----------------------

    max_iterations: int = 100
    """Maximum iterations per time step."""

    tolerance: float = 1e-6
    """Convergence absolute tolerance."""

    relaxation: float = 1.0
    """Relaxation factor ω (0 < ω ≤ 1)."""

    convergence_vars: list[str] = dc_field(default_factory=list)
    """Exchanged variables need convergence check.  
    If empty, all linked variables are checked."""

    divergence_action: DivergenceAction = DivergenceAction.ROLLBACK


@dataclass
class IterationResult:
    """Outcome of one LOOP iteration cycle."""

    converged: bool
    """Whether the iteration converged."""

    iterations: int
    """Number of iterations performed."""

    residual: float
    """Final residual value (max abs diff)."""

    residual_history: list[float] = dc_field(default_factory=list)
    """Per-iteration residual sequence."""

    message: str = ""
    """Any human-readable status message."""


# ---------------------------------------------------
# region ICouplingStrategy
# ---------------------------------------------------


class ICouplingStrategy(ABC):
    """Abstract strategy executing a coupling between components."""

    @property
    @abstractmethod
    def mode(self) -> str:
        """The coupling kind the strategy serves (CouplingKinds)."""
        ...

    @abstractmethod
    def execute(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ) -> IterationResult:
        """Execute the coupling cycle for the current time step."""
        ...


# ---------------------------------------------------
# region IPull/PushCoupler
# ---------------------------------------------------


class IPullCoupler(ICouplingStrategy):
    """One-way pull transfer through the provider's adapter chain.

    Default semantics: refresh() the chain, `input.pull()` into the
    consumer's buffer. Stateless; one instance may serve all PULL
    links of a composition.
    """

    @abstractmethod
    def transfer(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ):
        """Refresh adapters and move current values provider->consumer."""
        ...


class IPushCoupler(ICouplingStrategy):
    """One-way push transfer. Unlike PULL, the provider initiates the
    write; the consumer is not required to update() afterwards (sinks,
    loggers, live dashboards)."""

    @abstractmethod
    def push(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ):
        """Write the provider's current values into the consumer's
        input buffer through the adapter chain."""
        ...


# ---------------------------------------------------
# region IIterativeCoupler
# ---------------------------------------------------


class IIterativeCoupler(ICouplingStrategy):
    """Bidirectional iterative coupler (fixed-point).

    Subclasses implement domain-specific extraction/application of
    exchanged variables. Rollback snapshots come from IStateful.
    """

    @abstractmethod
    def iterate(
        self,
        comp_a: ILinkableModel,
        comp_b: ILinkableModel,
        config: CouplingConfig,
    ) -> IterationResult:
        """Run one fixed-point iteration cycle:

        1. Snapshot initial states of both components.
        2. For k = 1 .. max_iterations:
             a. Update comp_a (using comp_b's latest output as BC).
             b. Extract exchange variables from comp_a → comp_b.
             c. Update comp_b (using comp_a's latest output as source).
             d. Extract exchange variables from comp_b → comp_a.
             e. Check convergence; break if residual < tolerance.
             f. Apply relaxation if ω < 1.
        3. If not converged: apply divergence_action.
        """
        ...

    @abstractmethod
    def converge(
        self,
        previous: dict[str, ArrayLike],
        current: dict[str, ArrayLike],
        config: CouplingConfig,
    ) -> tuple[bool, float]:
        """(converged, residual); residual = max abs diff across all
        convergence variables."""
        ...


# ---------------------------------------------------
# region ISurrogateCoupler
# ---------------------------------------------------


class ISurrogateCoupler(ICouplingStrategy):
    """Surrogate coupling: an AI model stands in for a physics model.

    The contract is PORT ISOMORPHISM: the surrogate exposes the same
    quantity/element-set ports as the physics model it replaces, so
    the composition graph needs no rewiring when swapping them. The
    coupler verifies that isomorphism and executes plain transfers
    (surrogate links are usually PULL against the surrogate).
    """

    @abstractmethod
    def validate_ports(
        self,
        physics: ILinkableModel,
        surrogate: ILinkableModel,
    ) -> list[str]:
        """Check port isomorphism between the two models:
        same port ids, same quantities (units may differ only by a
        declared si_factor/si_offset), compatible element sets (or an
        adapter on the link). Empty list = swappable."""
        ...


# ---------------------------------------------------
# region IAgentCoupler
# ---------------------------------------------------


class IAgentCoupler(ICouplingStrategy):
    """Agent-driven control coupling: observe components, inject actions."""

    @abstractmethod
    def observe(
        self,
        agent: ILinkableModel,
        sources: list[ILinkableModel],
        config: CouplingConfig,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def apply_action(
        self,
        agent: ILinkableModel,
        target: ILinkableModel,
        action: Any,
        config: CouplingConfig,
    ): ...


# ---------------------------------------------------
# region INestedCoupler
# ---------------------------------------------------


class INestedCoupler(ICouplingStrategy):
    """Nested / multi-scale coupling: children run at finer time
    resolution and feed results back to the parent."""

    @abstractmethod
    def run_children(
        self,
        parent: ILinkableModel,
        children: list[ILinkableModel],
        config: CouplingConfig,
    ) -> IterationResult: ...

    @abstractmethod
    def aggregate(
        self,
        parent: ILinkableModel,
        children: list[ILinkableModel],
        config: CouplingConfig,
    ): ...
