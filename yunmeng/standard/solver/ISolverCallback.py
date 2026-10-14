# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Run-time callback protocols — mainly for the solver monitoring.

Design invariants of the monitoring layer:

1. Sensing only. A monitor observes and reports; ACTUATION (fallback,
   retraining, dt adaptation, abort) belongs to a supervisor consuming
   MonitorEvents (e.g. an OnlineLoop). A monitor's `action` field is a
   RECOMMENDATION, never an executed side effect.

2. Non-invasive under both modes. Monitors read through to_host() /
   detached views. Under TRAIN they must be graph-neutral: monitoring
   never extends the autograd graph and never forces synchronization
   beyond what the read itself requires. Monitors that cannot afford
   per-step cost must support a `stride` (check every k steps).

3. Hysteresis, not flicker. Threshold crossings use a high/low pair:
   an event fires when the metric crosses the HIGH threshold, and the
   alarm clears only when it falls below the LOW one. Monitors
   without hysteresis needs set the two equal.

4. SWMM-style accounting. Conservation monitors keep BOTH a per-step
   ledger and a run-total ledger, and report continuity error as a
   relative quantity — absolute residuals are meaningless across
   scales. report() returns a serializable dict suitable for CSV /
   run-summary output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from dataclasses import dataclass, field as dc_field
from enum import Enum

from yunmeng.standard.supports import IMesh

if TYPE_CHECKING:
    from yunmeng.standard.solver.ISolver import ISolver

# ---------------------------------------------------
# region ISolverCallback
# ---------------------------------------------------


class ISolverCallback(ABC):
    """Solver stepping hooks."""

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """The unique name of the callback."""
        ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def setup(self, solver: ISolver, mesh: IMesh, **kwargs): ...

    @abstractmethod
    def cleanup(self): ...

    @abstractmethod
    def on_task_begin(self): ...

    @abstractmethod
    def on_task_end(self): ...

    @abstractmethod
    def on_step_begin(self): ...

    @abstractmethod
    def on_step_end(self): ...


# ---------------------------------------------------
# region IMonitor
# ---------------------------------------------------


class MonitorSeverity(Enum):
    """How bad the observed condition is."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MonitorAction(Enum):
    """What the monitor RECOMMENDS the supervisor do."""

    NONE = "none"
    """Nothing; purely informational."""

    WARN = "warn"
    """Log and continue."""

    FALLBACK = "fallback"
    """Switch to the safe configuration 
    (e.g. physics operator)."""

    REDUCE_DT = "reduce_dt"
    """Tighten the time step(stability guardrail)."""

    ABORT = "abort"
    """Stop the run; the state is unrecoverable."""


@dataclass(frozen=True)
class MonitorEvent:
    """One emitted observation."""

    monitor_id: str
    time: float
    severity: MonitorSeverity
    action: MonitorAction
    message: str
    metrics: dict = dc_field(default_factory=dict)
    """The numbers behind the event (e.g. 
    {"continuity_error": 0.042,"courant_max": 1.31}), 
    so supervisors can act on values, not text."""


@dataclass
class Threshold:
    """High/low hysteresis pair on one metric.

    Fires when the metric rises above `high`;
    clears when it falls below `low`.
    Set low == high for a plain threshold.
    """

    high: float
    low: float = None

    def __post_init__(self):
        self.low = self.low or self.high
        if self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")


class IMonitor(ISolverCallback):
    """A solver callback that observes run health and emits events.

    Lifecycle: setup() binds solver/mesh/DataHub; on_step_end() is the
    canonical check point (fields and flux products of the step are
    complete); cleanup() flushes the ledger. The ISolverCallback
    on_step/on_step_begin hooks are available for monitors that need
    intra-step observations (rare).
    """

    # -- identity -----------------------------------

    @property
    @abstractmethod
    def kind(self) -> str:
        """Monitor family: "conservation" | "state" | "convergence" |
        "stability" | <open>."""
        ...

    # -- event surface ------------------------------

    @property
    @abstractmethod
    def events(self) -> list[MonitorEvent]:
        """All events emitted so far this run (oldest first)."""
        ...

    @abstractmethod
    def clear_events(self):
        """Drop emitted events (called by the supervisor after
        consuming them, or on solver reset())."""
        ...

    # -- reporting ----------------------------------

    @abstractmethod
    def report(self) -> dict:
        """SWMM-style run summary: serializable dict with the per-step
        and cumulative ledgers, extrema (with times), and the final
        status of each watched metric."""
        ...
