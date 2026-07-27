# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Core interfaces for the Solution layer — model lifecycle and topology.

Design principles:
  - Components are coarse-grained: a basin model, a hydrodynamic model,
    a water-quality module, etc. Each model may contain internal
    topology (sub-basins, grid cells, river reaches) that is opaque
    to the coupling framework.
  - Scheduler drives the simulation by calling update() on trigger
    components; data flows via PULL (one-way) or LOOP (iterative two-way).
  - Status machine enforces a strict lifecycle:
    CREATED → READY → RUNNING → DONE / FAILED.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
from dataclasses import dataclass, field, fields as dc_fields

from yunmeng.solutions.standards.IExchange import IInput, IOutput
from yunmeng.solutions.standards.ITopology import GeometryType

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
    """Model is actively computing (inside update())."""

    DONE = "done"
    """Model has completed its time horizon; further update() 
    calls are no-ops."""

    FAILED = "failed"
    """Irrecoverable errors occurred, and the model should be 
    finalized and re-instantiated."""


# ---------------------------------------------------
# region ModelMeta
# ---------------------------------------------------


@dataclass
class ExchangeMeta:
    """Exchange port item meta."""

    name: str
    description: str = ""
    quantity: str = ""  # e.g. "discharge", "water_level"
    unit: str = ""  # SI unit string, e.g. "m3/s"
    gtype: GeometryType = GeometryType.IDBASED
    temporal: str = "instant"  # instant, cumulative, ...
    dtype: str = "float64"
    required: bool = True


@dataclass
class ParamMeta:
    """Tunable parameter meta."""

    name: str
    description: str = ""
    dtype: str = "float"
    bounds: tuple = (None, None)  # (min, max)
    default: Any = None
    required: bool = False


@dataclass
class ModelMeta:
    """Basic model meta."""

    name: str = ""
    description: str = ""
    version: str = "1.0"
    category: str = ""

    outputs: list[ExchangeMeta] = []
    inputs: list[ExchangeMeta] = []
    parameters: list[ParamMeta] = []

    supports_loop: bool = False


# ---------------------------------------------------
# region ILinkableModel
# ---------------------------------------------------


class ILinkableModel(ABC):
    """Coarse-grained model model that can be linked with others.

    Lifecycle (driven by Scheduler):
        CREATED  → initialize() → READY
        READY    → prepare()    → READY   (idempotent, may be repeated)
        READY    → update()     → RUNNING → READY (if more steps)
                                          → DONE  (if end time reached)
                                          → FAILED (on error)
        DONE/FAILED → finish()  → CREATED (supports  re-initialization)
    """

    # -- class-level metadata -----------------------

    @classmethod
    @abstractmethod
    def get_meta(cls) -> ModelMeta:
        """Return metadata describing this model."""
        pass

    # -- instance properties ------------------------

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique instance identifier."""
        pass

    @property
    @abstractmethod
    def status(self) -> ModelStatus:
        """Current lifecycle state."""
        pass

    @property
    @abstractmethod
    def inputs(self) -> list[IInput]:
        """Input exchange ports."""
        pass

    @property
    @abstractmethod
    def outputs(self) -> list[IOutput]:
        """Output exchange ports."""
        pass

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
    def update(self) -> ModelStatus:
        """Advance the model by one logical time step.

        In PULL mode the Scheduler has already pushed data from upstream
        providers into input buffers before calling update().

        In LOOP mode the IterativeCoupler manages the exchange and
        convergence; update() performs a single inner solve.
        """
        pass

    @abstractmethod
    def finish(self):
        """Release resources, flush outputs, close files.

        After finish(), the model returns to CREATED and may be
        re-initialized for a new run."""
        pass
