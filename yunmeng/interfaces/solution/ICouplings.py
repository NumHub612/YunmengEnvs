# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Coupling strategy protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum

from yunmeng.interfaces.solution.IModel import ILinkableModel

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
# region ICoupler
# ---------------------------------------------------


class ICoupler(ABC):
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
