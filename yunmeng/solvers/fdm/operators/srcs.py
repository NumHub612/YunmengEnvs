# -*- encoding: utf-8 -*-
"""
Copyright (C) 2025, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Source term operators for the finite difference method.

Stage-2 refactor: the source function is applied VECTORIZED when possible
(source_func must accept (N, ...) coordinate/field arrays and return
(N, ...) values); falls back to the per-node loop otherwise. For TRAIN
mode, provide a tensorizable source_func — the per-node fallback breaks
the autograd graph.
"""

from yunmeng.solvers.interfaces import (
    IBoundaryCondition,
    IOperator,
    OperatorType,
)
from yunmeng.solvers.commons.solvers import BaseExplicitOperator, BaseImplicitOperator
from yunmeng.numerics.grids import Grid, ElementType
from yunmeng.numerics.fields import (
    DataHub,
    DataProduct,
    Sample,
    Field,
    Variable,
    VariableType,
)
from yunmeng.solvers.commons.supports import backend_of_field

from typing import Callable
import numpy as np


class Src01(BaseExplicitOperator):
    """Explicit source term operator on structured grids."""

    @classmethod
    def get_type(cls) -> OperatorType:
        return OperatorType.SRC

    @classmethod
    def get_name(cls) -> str:
        return "src01"

    def __init__(self, target_fields: list[str], tau: float, source_func: Callable):
        super().__init__(target_fields)

        self._var = target_fields[0]
        self._tau = tau
        self._source_func = source_func
        self._coords = None  # (N, dim) cached coordinates

    def prepare(
        self,
        mesh: Grid,
        bounds: dict[str, list[IBoundaryCondition]],
    ):
        super().prepare(mesh, bounds)
        if not isinstance(mesh, Grid) or not mesh.uniform:
            raise ValueError(f"FDM op {self.get_name()} only supports uniform Grid.")

        # Cache node coordinates for the vectorized path
        self._coords = np.array(
            [self._mesh.nodes[nid].coordinate for nid in range(self._mesh.node_count)]
        )

    def forward(self, datahub: DataHub, time: float) -> Field:
        sample = datahub.latest(self._var, ElementType.NODE)
        if sample is None:
            raise ValueError(f"Src01: no data for '{self._var}'@NODE in data_hub")
        old_field = sample.data
        backend = backend_of_field(old_field)
        new_field = Field(old_field.mesh_shards, old_field.vtype, old_field.etype)

        if len(old_field.mesh_shards) != 1:
            raise ValueError("FDM op src01 only supports single shard.")

        u = old_field._shards[0].data

        # Vectorized path (autograd-safe if source_func is tensorizable)
        try:
            coords = backend.asarray(self._coords, dtype=u.dtype)
            source_vals = self._tau * self._source_func(coords, u)
            new_field._shards[0].data = source_vals
        except (TypeError, ValueError, AttributeError):
            # Fallback: per-node loop (numpy-era behavior; breaks the graph —
            # do not rely on this path in TRAIN mode).
            for nid in range(self._mesh.node_count):
                coor = self._mesh.nodes[nid].coordinate
                source_val = self._tau * self._source_func(coor, old_field[nid])
                new_field[nid] = source_val

        self._publish(
            datahub,
            self._var,
            ElementType.NODE,
            Sample(time, new_field),
            self.get_type().value,
        )
        return new_field
