# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

description
"""

from abc import ABC, abstractmethod
from typing import Any

from yunmeng.interfaces.types import ArrayLike
from yunmeng.interfaces.solution import ICoupler, CouplingConfig, IterationResult
from yunmeng.solutions.commons.models import BaseModel

# ---------------------------------------------------
# region IPull/PushCoupler
# ---------------------------------------------------


class IPullCoupler(ICoupler):
    """One-way pull transfer through the provider's adapter chain.

    Default semantics: refresh() the chain, `input.pull()` into the
    consumer's buffer. Stateless; one instance may serve all PULL
    links of a composition.
    """

    @abstractmethod
    def transfer(
        self,
        source: BaseModel,
        target: BaseModel,
        config: CouplingConfig,
    ):
        """Refresh adapters and move current values provider->consumer."""
        ...


class IPushCoupler(ICoupler):
    """One-way push transfer. Unlike PULL, the provider initiates the
    write; the consumer is not required to update() afterwards (sinks,
    loggers, live dashboards)."""

    @abstractmethod
    def push(
        self,
        source: BaseModel,
        target: BaseModel,
        config: CouplingConfig,
    ):
        """Write the provider's current values into the consumer's
        input buffer through the adapter chain."""
        ...


# ---------------------------------------------------
# region IIterativeCoupler
# ---------------------------------------------------


class IIterativeCoupler(ICoupler):
    """Bidirectional iterative coupler (fixed-point).

    Subclasses implement domain-specific extraction/application of
    exchanged variables. Rollback snapshots come from IStateful.
    """

    @abstractmethod
    def iterate(
        self,
        comp_a: BaseModel,
        comp_b: BaseModel,
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


class ISurrogateCoupler(ICoupler):
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
        physics: BaseModel,
        surrogate: BaseModel,
    ) -> list[str]:
        """Check port isomorphism between the two models:
        same port ids, same quantities (units may differ only by a
        declared si_factor/si_offset), compatible element sets (or an
        adapter on the link). Empty list = swappable."""
        ...


# ---------------------------------------------------
# region IAgentCoupler
# ---------------------------------------------------


class IAgentCoupler(ICoupler):
    """Agent-driven control coupling: observe components, inject actions."""

    @abstractmethod
    def observe(
        self,
        agent: BaseModel,
        sources: list[BaseModel],
        config: CouplingConfig,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def apply_action(
        self,
        agent: BaseModel,
        target: BaseModel,
        action: Any,
        config: CouplingConfig,
    ): ...


# ---------------------------------------------------
# region INestedCoupler
# ---------------------------------------------------


class INestedCoupler(ICoupler):
    """Nested / multi-scale coupling: children run at finer time
    resolution and feed results back to the parent."""

    @abstractmethod
    def run_children(
        self,
        parent: BaseModel,
        children: list[BaseModel],
        config: CouplingConfig,
    ) -> IterationResult: ...

    @abstractmethod
    def aggregate(
        self,
        parent: BaseModel,
        children: list[BaseModel],
        config: CouplingConfig,
    ): ...
