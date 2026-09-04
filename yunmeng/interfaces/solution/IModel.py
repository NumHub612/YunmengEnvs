# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Model-layer protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import Any

from yunmeng.interfaces.solution.IDataset import IElementSet, Quantity
from yunmeng.interfaces.solution.IExchange import IInput, IOutput
from yunmeng.interfaces.types import GeometryType, ParamMeta

# ---------------------------------------------------
# region ModelStatus
# ---------------------------------------------------


class ModelStatus(Enum):
    """Lifecycle states for a linkable model."""

    CREATED = "created"
    """Model instance has been constructed but not initialized."""

    READY = "ready"
    """Initialization and preparation have succeeded;the model 
    can participate in update cycles."""

    RUNNING = "running"
    """Model is actively computing."""

    DONE = "done"
    """Model has completed its time horizon; further update() 
    calls are no-ops."""

    FAILED = "failed"
    """Irrecoverable errors occurred, and the model should be 
    finalized and re-instantiated."""


# ---------------------------------------------------
# region Meta value types
# ---------------------------------------------------


@dataclass
class ExchangeMeta:
    """Exchange port item meta."""

    name: str
    description: str = ""
    quantity: str = ""  # e.g. "discharge", "water_level"
    unit: str = ""  # SI unit string, e.g. "m3/s"
    gtype: GeometryType = GeometryType.POINT
    temporal: str = "instant"  # instant, cumulative, ...
    dtype: str = "float64"
    required: bool = True


@dataclass
class ModelMeta:
    """Basic model meta."""

    name: str = ""
    description: str = ""
    version: str = "1.0"
    category: str = ""

    outputs: list[ExchangeMeta] = dc_field(default_factory=list)
    inputs: list[ExchangeMeta] = dc_field(default_factory=list)
    parameters: list[ParamMeta] = dc_field(default_factory=list)

    supports_loop: bool = False


# ---------------------------------------------------
# region Callbacks
# ---------------------------------------------------


class CallbackEvent:
    """Standard lifecycle event names."""

    BEFORE_INITIALIZE = "before_initialize"
    AFTER_INITIALIZE = "after_initialize"
    ON_PREPARE = "on_prepare"
    BEFORE_UPDATE = "before_update"
    AFTER_UPDATE = "after_update"
    ON_FINISH = "on_finish"
    ON_ERROR = "on_error"

    STEP_BEGIN = "step_begin"
    STEP_END = "step_end"


class ICallback(ABC):
    """Plugin invoked at model lifecycle points."""

    @abstractmethod
    def on_event(
        self,
        event: str,
        model: "ILinkableModel",
        context: dict,
    ): ...


# ---------------------------------------------------
# region IStateful
# ---------------------------------------------------


class IStateful(ABC):
    """Interface for components whose full internal state can be
    captured to an in-memory snapshot and restored later.

    Snapshots are opaque to the framework; only the component itself
    knows how to serialize and deserialize them.  They should be
    treated as immutable blobs — the framework never modifies them.
    """

    @abstractmethod
    def snapshot(self) -> Any:
        """Capture a complete, self-contained snapshot of the
        component's current internal state.

        The returned object must contain everything needed to bring
        the component back to exactly this state: field values,
        parameter sets, time counters, random seeds, etc.
        """
        pass

    @abstractmethod
    def restore(self, snapshot: Any):
        """Restore the component to the exact state captured in
        *snapshot*.

        After restore(), the component must behave as if it had just
        completed the update() call that produced this snapshot.
        Status should be set to READY.
        """
        pass

    def diff(self, snapshot_a: Any, snapshot_b: Any) -> dict:
        """Compare two snapshots and return a human-readable diff.

        Default implementation returns an empty dict; override to
        provide domain-specific diagnostics.
        """
        return {}


# ---------------------------------------------------
# region ILinkableModel
# ---------------------------------------------------


class ILinkableModel(ABC):
    """Coarse-grained linkable model. UNCHANGED state machine and port
    semantics (v2.0 §8: the base protocol carries no estimation or
    mode members).

    Lifecycle (driven by Scheduler):
        CREATED  → initialize() → READY
        READY    → prepare()    → READY   (idempotent)
        READY    → update()     → RUNNING → READY / DONE / FAILED
        DONE/FAILED → finish()  → CREATED (re-initializable)
    """

    # -- class-level metadata -----------------------

    @classmethod
    @abstractmethod
    def get_meta(cls) -> ModelMeta: ...

    # -- instance properties ------------------------

    @property
    @abstractmethod
    def id(self) -> str: ...

    @property
    @abstractmethod
    def status(self) -> ModelStatus: ...

    @property
    @abstractmethod
    def callbacks(self) -> list[ICallback]: ...

    @property
    @abstractmethod
    def inputs(self) -> list[IInput]: ...

    @property
    @abstractmethod
    def outputs(self) -> list[IOutput]: ...

    # -- assemble -----------------------------------

    @abstractmethod
    def remove_callback(self, cb: ICallback): ...

    @abstractmethod
    def add_callback(self, cb: ICallback): ...

    @abstractmethod
    def get_output(self, port_id: str) -> IOutput: ...

    @abstractmethod
    def get_input(self, port_id: str) -> IInput: ...

    @abstractmethod
    def create_input(
        self,
        quantity: Quantity,
        elements: IElementSet = None,
        port_id: str = None,
    ) -> IInput: ...

    @abstractmethod
    def create_output(
        self,
        quantity: Quantity,
        elements: IElementSet = None,
        port_id: str = None,
    ) -> IOutput: ...

    @abstractmethod
    def remove_port(self, port_id: str) -> bool:
        """Remove a port, disconnecting from the coupling graph first
        (providers detached, consumers dropped)."""
        ...

    # -- lifecycle ----------------------------------

    @abstractmethod
    def initialize(self):
        """Initialize internal data structures, load parameters, build
        internal topology.  After success, status must be READY."""
        pass

    @abstractmethod
    def validate(self) -> list[str]:
        """Validate configuration and return a list of error messages.
        An empty list means validation passed."""
        pass

    @abstractmethod
    def prepare(self):
        """Pre-allocate buffers, warm caches, finalize compute graphs.
        Called once before the first update() in a run."""
        pass

    @abstractmethod
    def update(self, inquirers: list[IOutput] = None) -> ModelStatus:
        """Advance the model by one logical time step.

        In PULL mode `Scheduler` has already pushed data from upstream
        providers into input buffers before calling update().
        In LOOP mode the IterativeCoupler manages the exchange and
        convergence; update() performs a single inner solve.
        """
        pass

    @abstractmethod
    def finish(self):
        """Release resources, flush outputs, close files.
        After called finish(), the model returns to CREATED and may be
        re-initialized for a new run."""
        pass
