# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Coupling protocol interfaces — PULL (one-way) and LOOP (iterative two-way).

The Scheduler uses these interfaces to manage how data flows between
LinkableComponents:

  PULL mode (default):
      Scheduler calls trigger.update() which pulls data from upstream
      outputs through linked inputs.  Data flows one direction only.

  LOOP mode (for bidirectional feedback):
      Scheduler delegates to an IIterativeCoupler which repeatedly
      updates two connected components until their exchanged variables
      converge (fixed-point iteration with optional relaxation).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from yunmeng.solutions.standards.IModel import ILinkableModel

# ---------------------------------------------------
# region CouplingMode
# ---------------------------------------------------


class CouplingMode(Enum):
    """How two or more components are coupled."""

    PULL = "pull"
    """One-way data pull (OpenMI-style).  Downstream Model reads
    from upstream output when it updates."""

    LOOP = "loop"
    """Iterative two-way coupling. Both components exchange data and
    iterate within a single time step until convergence."""

    PUSH = "push"
    """One-way data push (rarely used in practice)."""

    AGENT = "agent"
    """Agent-driven control coupling. An ``IAgentModel`` observes one or
    more components and injects actions back into the system as boundary
    conditions or source terms."""

    SURROGATE = "surrogate"
    """Surrogate coupling. A fast AI model replaces or accelerates an
    expensive physics model while keeping the same exchange ports."""

    NESTED = "nested"
    """Nested coupling. A parent model spawns one or more child models
    that run with their own time step and feed results back to the parent."""

    HYBRID = "hybrid"
    """Hybrid coupling. Combines physics and AI models in a single step,
    for example a neural-network corrector applied after a PDE solver."""


# ---------------------------------------------------
# region CouplingConfig
# ---------------------------------------------------


@dataclass
class CouplingConfig:
    """Configuration for a coupling link."""

    mode: CouplingMode = CouplingMode.PULL

    # -- LOOP-only parameters -----------------------

    max_iterations: int = 100
    """Maximum iterations per time step."""

    tolerance: float = 1e-6
    """Convergence tolerance (max absolute difference)."""

    relaxation: float = 1.0
    """Relaxation factor ω (0 < ω ≤ 1)."""

    convergence_vars: list[str] = field(default_factory=list)
    """Exchanged variables need convergence check.  
    If empty, all linked variables are checked."""

    divergence_action: str = "rollback"
    """Action when max_iterations is reached without convergence:
    "rollback" — restore pre-iteration state and mark FAILED 
    "continue" — accept the best approximation and emit a warning
    "freeze"   — keep the last converged state from previous step
    """


# ---------------------------------------------------
# region IterationResult
# ---------------------------------------------------


@dataclass
class IterationResult:
    """Outcome of one LOOP iteration cycle."""

    converged: bool
    """Whether the iteration converged within tolerance."""

    iterations: int
    """Number of iterations actually performed."""

    residual: float
    """Final residual value (max abs diff)."""

    residual_history: list[float] = field(default_factory=list)
    """Per-iteration residual sequence."""

    message: str = ""
    """Human-readable status message."""


# ---------------------------------------------------
# region CouplingStrategy
# ---------------------------------------------------


class ICouplingStrategy(ABC):
    """Abstract strategy for executing a coupling between two or more
    components.  The Scheduler selects the appropriate strategy based
    on the CouplingMode declared in the link configuration."""

    @property
    @abstractmethod
    def mode(self) -> CouplingMode:
        pass

    @abstractmethod
    def execute(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ) -> IterationResult:
        """Executes coupling cycle for this current time step.

        For PULL this is a trivial data transfer.
        For LOOP this performs the full fixed-point iteration.
        """
        pass


# ---------------------------------------------------
# region IterativeCoupler
# ---------------------------------------------------


class IIterativeCoupler(ICouplingStrategy):
    """Bidirectional iterative coupler using fixed-point iteration.

    Concrete subclasses implement the domain-specific logic for
    extracting exchanged variables from one Model and applying
    them as boundary conditions / source terms to the other.
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
        pass

    @abstractmethod
    def converge(
        self,
        previous: dict[str, np.ndarray],
        current: dict[str, np.ndarray],
        config: CouplingConfig,
    ) -> tuple[bool, float]:
        """Return (converged, residual) where residual is the maximum
        absolute difference across all convergence variables."""
        pass


# ---------------------------------------------------
# region Specialized coupling strategies
# ---------------------------------------------------


class IAgentCouplingStrategy(ICouplingStrategy):
    """Coupling strategy driven by an ``IAgentModel``.

    The agent observes one or more components, selects an action, and the
    action is applied as an input to a downstream target component.
    """

    @abstractmethod
    def observe(
        self,
        agent: ILinkableModel,
        sources: list[ILinkableModel],
        config: CouplingConfig,
    ) -> dict[str, Any]:
        """Collect observations from all source components."""
        pass

    @abstractmethod
    def apply_action(
        self,
        agent: ILinkableModel,
        target: ILinkableModel,
        action: Any,
        config: CouplingConfig,
    ):
        """Write the action values into the target's input ports."""
        pass


class ISurrogateCouplingStrategy(ICouplingStrategy):
    """Strategy for surrogate coupling.

    Decides when to call the expensive physics model vs. the fast
    surrogate, and synchronizes their states.
    """

    @abstractmethod
    def should_use_surrogate(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ) -> bool:
        """Return True if the surrogate should be used for this step."""
        pass

    @abstractmethod
    def synchronize(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ):
        """Synchronize the surrogate with the ground-truth model."""
        pass


class INestedCouplingStrategy(ICouplingStrategy):
    """Strategy for nested / multi-scale coupling.

    Manages a parent model and one or more child models that run at
    different time resolutions and feed results back to the parent.
    """

    @abstractmethod
    def run_children(
        self,
        parent: ILinkableModel,
        children: list[ILinkableModel],
        config: CouplingConfig,
    ) -> IterationResult:
        """Advance child models over the parent's current step."""
        pass

    @abstractmethod
    def aggregate(
        self,
        parent: ILinkableModel,
        children: list[ILinkableModel],
        config: CouplingConfig,
    ):
        """Aggregate child outputs into parent inputs."""
        pass


class IHybridCouplingStrategy(ICouplingStrategy):
    """Strategy for hybrid AI + physics coupling.

    Runs the physics model first, then applies an AI correction to the
    exchanged variables (or vice versa).
    """

    @abstractmethod
    def physics_step(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ):
        """Execute the physics-model part of the hybrid step."""
        pass

    @abstractmethod
    def ai_correction(
        self,
        source: ILinkableModel,
        target: ILinkableModel,
        config: CouplingConfig,
    ):
        """Apply the AI correction to the exchanged variables."""
        pass
