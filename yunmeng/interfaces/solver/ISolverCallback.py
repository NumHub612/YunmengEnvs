# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solver callback interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from yunmeng.interfaces.support.mesh import IMesh

if TYPE_CHECKING:
    from yunmeng.interfaces.solver.ISolver import ISolver

# ---------------------------------------------------
# region Callback / equation
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
    def on_step(self): ...

    @abstractmethod
    def on_step_end(self): ...
