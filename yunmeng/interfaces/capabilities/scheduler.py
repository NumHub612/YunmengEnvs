# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Composition and scheduling protocols — the solution layer's driver.

A *composition* is the coupling graph of one scenario: models (nodes)
plus directed links (edges, each carrying a CouplingConfig). A
*scheduler* drives the composition through time: it owns the global
clock and invokes model lifecycles in an order derived from the graph.

Design invariants of the composition layer:

1. The graph is explicit, never implicit. A link exists only as a
   LinkSpec registered in the composition; port-to-port wiring outside
   the composition (direct `input.provider = output`) is illegal in
   scheduled runs.

2. Direction: every link points provider -> consumer. PULL and PUSH
   links must form a DAG. Cycles are legal ONLY when every edge of the
   cycle is LOOP kind; the scheduler treats each LOOP cycle as one
   super-node driven by an IIterativeCoupler.

3. The scheduler, not the models, owns time. Models never advance
   their own clocks; update() performs exactly one logical step of the
   model's own TimeSpan and returns. Global time alignment is checked
   at validate() (mismatched TimeSpan.step requires an explicit
   temporal adapter on the link — silent resampling stays forbidden).

4. TRAIN is forbidden composition-wide. The scheduler asserts EVAL on
   every model before the first step (see IModel invariant 2).

5. Engine-agnostic. The scheduler talks to ILinkableModel ports only;
   whether a model is solver-hosted or self-contained is invisible
   here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dc_field

from yunmeng.interfaces.solution.ICouplings import CouplingConfig, CouplingKinds
from yunmeng.interfaces.solution.IModel import ILinkableModel

# ---------------------------------------------------
# region Link spec
# ---------------------------------------------------


@dataclass
class PortRef:
    """One end of a link: a port on a model."""

    model_id: str
    port_id: str


@dataclass
class LinkSpec:
    """One directed coupling edge: provider port -> consumer port."""

    link_id: str
    source: PortRef
    """Provider side (an IOutput of source.model_id)."""

    target: PortRef
    """Consumer side (an IInput of target.model_id)."""

    kind: str = CouplingKinds.PULL
    """CouplingKinds constant; governs which strategy executes it."""

    config: CouplingConfig = dc_field(default_factory=CouplingConfig)
    """Execution parameters (LOOP iteration controls etc.)."""


# ---------------------------------------------------
# region IComposition
# ---------------------------------------------------


class IComposition(ABC):
    """The coupling graph of a scenario. Pure topology + validation;
    owns no runtime state."""

    # -- assemble -----------------------------------

    @abstractmethod
    def add_model(self, model: ILinkableModel): ...

    @abstractmethod
    def remove_model(self, model_id: str):
        """Remove the model and all links touching it."""
        ...

    @abstractmethod
    def add_link(self, link: LinkSpec):
        """Register a link; wires input.provider through the adapter
        chain at schedule time, not here."""
        ...

    @abstractmethod
    def remove_link(self, link_id: str): ...

    # -- introspection ------------------------------

    @property
    @abstractmethod
    def models(self) -> dict[str, ILinkableModel]: ...

    @property
    @abstractmethod
    def links(self) -> dict[str, LinkSpec]: ...

    @abstractmethod
    def links_of(self, model_id: str) -> list[LinkSpec]:
        """All links where the model is source or target."""
        ...

    # -- validation ---------------------------------

    @abstractmethod
    def validate(self) -> list[str]:
        """Static graph checks. Must report at least:

        - dangling port refs (model/port does not exist);
        - kind/port-direction mismatch (source not an IOutput, etc.);
        - illegal cycles (any cycle containing a non-LOOP edge);
        - quantity/unit mismatch without an adapter on the link;
        - TimeSpan.step mismatch without a temporal adapter;
        - more than one provider wired to the same IInput.
        """
        ...


# ---------------------------------------------------
# region IScheduler
# ---------------------------------------------------


class IScheduler(ABC):
    """Drives a composition through its time horizon.

    Execution model per global step:
      1. Resolve execution order: topological order of the DAG obtained
         by collapsing every LOOP cycle into one super-node.
      2. For each PULL/PUSH/SURROGATE link in order: refresh adapter
         chain, transfer values provider -> consumer.
      3. For each LOOP super-node: delegate to the IIterativeCoupler
         until convergence or divergence_action.
      4. Call update() on each model in order.
      5. Advance the global clock; emit scheduler events.

    The scheduler asserts EVAL mode on all models before step 1 of the
    first global step.
    """

    @abstractmethod
    def attach(self, composition: IComposition):
        """Bind a composition; validates it and raises on ERROR issues."""
        ...

    @abstractmethod
    def execution_order(self) -> list[str]:
        """Resolved model order (LOOP cycles appear as one entry:
        the coupler id)."""
        ...

    @abstractmethod
    def initialize(self):
        """initialize() + prepare() all models in execution order."""
        ...

    @abstractmethod
    def step(self) -> bool:
        """Run one global step. Returns False when the horizon is
        exhausted (all models DONE)."""
        ...

    @abstractmethod
    def run(self):
        """initialize(); step() until done; finish() all models."""
        ...

    @property
    @abstractmethod
    def current_time(self) -> float: ...

    @abstractmethod
    def finish(self):
        """finish() all models (flush IO, release resources), even
        after a FAILED step."""
        ...
