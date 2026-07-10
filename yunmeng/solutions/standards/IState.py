# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Stateful component interface.

Provides snapshot/restore capability for components  need to
save and later revert their internal state.  Used by:
  - the IterativeCoupler to implement rollback on divergence
  - the Scheduler to support breakpoint-and-resume workflows
  - long-running ensembles a branch from a common warm state
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


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
