# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Snapshottable-layer contracts.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------
# region ISnapshottable
# ---------------------------------------------------


@runtime_checkable
class ISnapshottable(Protocol):
    """Full-state persistence, two media, ONE contract.

    In-memory pair (snapshot/restore): hot paths — iterative coupler
    rollback, trial branching, hotstart.
    On-disk pair (save/load): cold paths — checkpoint files, resumes
    across processes. Default implementation pickles the in-memory
    snapshot; override for custom formats.

    Applicable layers: solver, model.
    """

    # -- in-memory -------------------------------

    @abstractmethod
    def snapshot(self) -> Any:
        """Capture a complete, self-contained snapshot."""
        ...

    @abstractmethod
    def restore(self, snapshot: Any):
        """Restore the exact state captured by snapshot."""
        ...

    # -- on-disk ---------------------------------

    def save(self, path: str):
        """Default: pickle the in-memory snapshot."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self.snapshot(), f)

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ISnapshottable": ...
