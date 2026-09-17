# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Model-layer protocols.

A Model describes a SCENARIO (e.g. a hydrodynamic simulation problem);
a Solver provides the COMPUTE ENGINE (e.g. a Saint-Venant or
advection-diffusion solver). A Model OWNS its Solver and delegates
stepping to it.

Design invariants of the model layer:

1. INTEGRABILITY is the model layer's concern; DIFFERENTIABILITY is the
   solver layer's. Exchange items (IInput/IOutput/IValueSet) carry COPY
   semantics only — the model boundary is an explicit gradient cut.
   Cross-component joint optimization goes through the copy-semantic
   parameter channel (IEstimable), never through autograd across ports.

2. TRAIN mode is STANDALONE-ONLY. set_mode(TRAIN) must raise if the
   model participates in a coupling graph (any input connected, or the
   model is held by a coupler). Coupling strategies must assert EVAL on
   both ends before execute(). All coupling kinds are EVAL-only
   ("train first, link after").

3. Data wiring: Model.update() pulls from inputs -> writes solver
   fields/boundaries -> solver.step(dt) -> reads solution fields ->
   publishes outputs. Spatial mapping: MESH-type ports reference solver
   Regions directly (zero-copy); POINT-type ports interpolate via
   ISpatialIndex. Temporal mismatches between TimeSpan.step and the
   solver's time step are resolved by adapters, never by silent
   resampling.

4. Namespaces: parameters exposed by a composed model are prefixed
   "<model_id>.<param>" (see split_namespaces); mode propagation runs
   model -> solver -> DataHub -> operators (propagate_mode); snapshots
   aggregate the solver state plus port buffers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field
from enum import Enum

from yunmeng.interfaces.solution.IDataset import IElementSet, Quantity
from yunmeng.interfaces.solution.IExchange import IInput, IOutput, ExchangeMeta
from yunmeng.interfaces.capabilities import ParamMeta

# ---------------------------------------------------
# region Status & Metas
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


class ModelEvent:
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


class IModelCallback(ABC):
    """Plugin invoked at model lifecycle points."""

    @abstractmethod
    def on_event(self, event: str, model: "ILinkableModel", context: dict): ...


# ---------------------------------------------------
# region ILinkableModel
# ---------------------------------------------------


class ILinkableModel(ABC):
    """Coarse-grained linkable model. The base protocol carries no
    estimation or mode members; optional capabilities (ISnapshottable,
    IAssimilatable, IParameterized via IEstimable) come from
    capabilities.py and are isinstance-governed.

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
    def outputs(self) -> list[IOutput]: ...

    @property
    @abstractmethod
    def inputs(self) -> list[IInput]: ...

    @property
    @abstractmethod
    def callbacks(self) -> list[IModelCallback]: ...

    @property
    @abstractmethod
    def status(self) -> ModelStatus: ...

    @property
    @abstractmethod
    def id(self) -> str: ...

    # -- assemble -----------------------------------

    @abstractmethod
    def remove_callback(self, cb: IModelCallback): ...

    @abstractmethod
    def add_callback(self, cb: IModelCallback): ...

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
        ...

    @abstractmethod
    def validate(self) -> list[str]:
        """Validate configuration and return a list of error messages.
        An empty list means validation passed."""
        ...

    @abstractmethod
    def prepare(self):
        """Pre-allocate buffers, warm caches, finalize compute graphs.
        Called once before the first update() in a run."""
        ...

    @abstractmethod
    def update(self, inquirers: list[IOutput] = None) -> ModelStatus:
        """Advance the model by one logical time step.

        In PULL mode `Scheduler` has already pushed data from upstream
        providers into input buffers before calling update().
        In LOOP mode the IterativeCoupler manages the exchange and
        convergence; update() performs a single inner solve.
        """
        ...

    @abstractmethod
    def finish(self):
        """Release resources, flush outputs, close files.
        After called finish(), the model returns to CREATED and may be
        re-initialized for a new run."""
        ...
