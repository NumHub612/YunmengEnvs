# -*- encoding: utf-8 -*-
"""
Copyright (C) 2026, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Solver registry — auto-discovery and factory loading for solver classes.

Usage:
    @SolverRegistry.register
    class MySolver(BaseSolver): ...

    # Later — load from saved state
    solver = SolverRegistry.load(path, mesh, operators)
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from yunmeng.numerics.mesh import Mesh
    from yunmeng.solvers.interfaces import ISolver, IOperator


class SolverRegistry:
    """
    Central registry mapping solver names (``get_name()``) to classes.

    Thread-safe for read operations. Registration should happen at
    import time (via decorator) and is not thread-safe by design.
    """

    _solvers: dict[str, type] = {}

    @classmethod
    def register(cls, solver_class: type) -> type:
        """
        Decorator: register a solver class.

        Example::

            @SolverRegistry.register
            class BurgersExplicitSolver(BaseSolver): ...
        """
        name = solver_class.get_name()
        if name in cls._solvers and cls._solvers[name] is not solver_class:
            raise RuntimeError(
                f"Solver name collision: '{name}' already registered to "
                f"{cls._solvers[name].__module__}.{cls._solvers[name].__qualname__}"
            )
        cls._solvers[name] = solver_class
        return solver_class

    @classmethod
    def get(cls, name: str) -> type:
        """Look up a solver class by name."""
        if name not in cls._solvers:
            known = ", ".join(sorted(cls._solvers.keys()))
            raise KeyError(f"Unknown solver '{name}'. Known: {known}")
        return cls._solvers[name]

    @classmethod
    def list_solvers(cls) -> dict[str, str]:
        """Return {name: qualified_class_name} for all registered solvers."""
        return {
            name: f"{sc.__module__}.{sc.__qualname__}"
            for name, sc in cls._solvers.items()
        }

    # ---- save / load -------------------------------------------------------

    @classmethod
    def save(cls, solver: ISolver, path: str) -> None:
        """
        Save solver state + metadata for later recovery.

        Writes two files:
            {path}.meta  — JSON with solver name, id, config
            {path}.npz   — NumPy arrays with field data
        """
        meta = {
            "solver_name": solver.get_name(),
            "id": solver.id,
            "config": solver.config.to_dict(),
            "status": {
                "current_time": solver.status.current_time,
                "end_time": solver.status.end_time,
                "time_step": solver.status.time_step,
                "steps": solver.status.steps,
                "iters": solver.status.iters,
            },
        }
        meta_path = path + ".meta"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save field arrays
        arrays = {}
        for name, field in solver._fields.items():
            for i, shard in enumerate(field._shards):
                arrays[f"{name}_shard{i}"] = shard.data
        if arrays:
            npz_path = path + ".npz"
            import numpy as np

            np.savez_compressed(npz_path, **arrays)

    @classmethod
    def load(
        cls, path: str, mesh: Mesh, operators: list[IOperator] | None = None
    ) -> ISolver:
        """
        Restore a solver from previously saved state.

        Args:
            path: Base path (without .meta/.npz extension).
            mesh: Mesh object — must be compatible with the saved solver.
            operators: Operator list — must match the saved solver's operators.

        Returns:
            Fully initialized solver with restored fields and status.
        """
        from yunmeng.solvers.interfaces import SolverConfig

        meta_path = path + ".meta"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        solver_class = cls.get(meta["solver_name"])

        # Reconstruct config
        config_class = solver_class.get_config_class()
        config = config_class.from_dict(meta.get("config", {}))

        # Instantiate
        solver = solver_class(meta["id"], mesh, operators, config=config)

        # Restore status
        st = meta.get("status", {})
        solver._status.current_time = st.get("current_time", 0.0)
        solver._status.end_time = st.get("end_time", None)
        solver._status.time_step = st.get("time_step", None)
        solver._status.steps = st.get("steps", 0)
        solver._status.iters = st.get("iters", 0)

        # Restore fields
        npz_path = path + ".npz"
        if os.path.exists(npz_path):
            import numpy as np

            data = np.load(npz_path)
            for name in solver._fields:
                for i, shard in enumerate(solver._fields[name]._shards):
                    key = f"{name}_shard{i}"
                    if key in data:
                        shard.data = data[key]

        return solver


# ---- convenience decorator alias ----
register_solver = SolverRegistry.register
